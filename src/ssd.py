"""SSD net based on MobileNetV2 backbone."""
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
import mindspore.ops as ops
from .fpn import mobilenet_v1_fpn, resnet50_fpn
from .vgg16 import vgg16
from .mobilenet_v1 import mobilenet_v1_Feature


def _make_divisible(v, divisor, min_value=None):
    """Ensures that all layers have a channel number that is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='same'):
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                     padding=0, pad_mode=pad_mod, has_bias=True)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-3, momentum=0.97,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _last_conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='same', pad=0):
    in_channels = in_channel
    out_channels = in_channel
    depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same',
                               padding=pad, group=in_channels)
    conv = _conv2d(in_channel, out_channel, kernel_size=1)
    return nn.SequentialCell([depthwise_conv, _bn(in_channel), nn.ReLU6(), conv])


class ConvBNReLU(nn.Cell):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, shared_conv=None):
        super(ConvBNReLU, self).__init__()
        padding = 0
        in_channels = in_planes
        out_channels = out_planes
        if shared_conv is None:
            if groups == 1:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same', padding=padding)
            else:
                out_channels = in_planes
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same',
                                 padding=padding, group=in_channels)
            layers = [conv, _bn(out_planes), nn.ReLU6()]
        else:
            layers = [shared_conv, _bn(out_planes), nn.ReLU6()]
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class InvertedResidual(nn.Cell):
    """
    Residual block definition for MobileNetV2.
    """
    def __init__(self, inp, oup, stride, expand_ratio, last_relu=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False),
            _bn(oup),
        ])
        self.conv = nn.SequentialCell(layers)
        self.cast = ops.Cast()
        self.last_relu = last_relu
        self.relu = nn.ReLU6()

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            x = identity + x
        if self.last_relu:
            x = self.relu(x)
        return x


class FlattenConcat(nn.Cell):
    """
    Concatenate predictions into a single tensor.
    """
    def __init__(self, config):
        super(FlattenConcat, self).__init__()
        self.num_ssd_boxes = config.num_ssd_boxes
        self.concat = ops.Concat(axis=1)
        self.transpose = ops.Transpose()
    def construct(self, inputs):
        output = ()
        batch_size = ops.shape(inputs[0])[0]
        for x in inputs:
            x = self.transpose(x, (0, 2, 3, 1))
            output += (ops.reshape(x, (batch_size, -1)),)
        res = self.concat(output)
        return ops.reshape(res, (batch_size, self.num_ssd_boxes, -1))


class MultiBox(nn.Cell):
    """
    Multibox conv layers for localization and classification.
    """
    def __init__(self, config):
        super(MultiBox, self).__init__()
        num_classes = config.num_classes
        out_channels = config.extras_out_channels
        num_default = config.num_default

        loc_layers = []
        cls_layers = []
        for k, out_channel in enumerate(out_channels):
            loc_layers += [_last_conv2d(out_channel, 4 * num_default[k],
                                        kernel_size=3, stride=1, pad_mod='same', pad=0)]
            cls_layers += [_last_conv2d(out_channel, num_classes * num_default[k],
                                        kernel_size=3, stride=1, pad_mod='same', pad=0)]

        self.multi_loc_layers = nn.CellList(loc_layers)
        self.multi_cls_layers = nn.CellList(cls_layers)
        self.flatten_concat = FlattenConcat(config)

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            loc_outputs += (self.multi_loc_layers[i](inputs[i]),)
            cls_outputs += (self.multi_cls_layers[i](inputs[i]),)
        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


class WeightSharedMultiBox(nn.Cell):
    """
    Weight shared Multi-box conv layers.
    """
    def __init__(self, config, loc_cls_shared_addition=False):
        super(WeightSharedMultiBox, self).__init__()
        num_classes = config.num_classes
        out_channels = config.extras_out_channels[0]
        num_default = config.num_default[0]
        num_features = len(config.feature_size)
        num_addition_layers = config.num_addition_layers
        self.loc_cls_shared_addition = loc_cls_shared_addition

        if not loc_cls_shared_addition:
            loc_convs = [
                _conv2d(out_channels, out_channels, 3, 1) for x in range(num_addition_layers)
            ]
            cls_convs = [
                _conv2d(out_channels, out_channels, 3, 1) for x in range(num_addition_layers)
            ]
            addition_loc_layer_list = []
            addition_cls_layer_list = []
            for _ in range(num_features):
                addition_loc_layer = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, loc_convs[x]) for x in range(num_addition_layers)
                ]
                addition_cls_layer = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, cls_convs[x]) for x in range(num_addition_layers)
                ]
                addition_loc_layer_list.append(nn.SequentialCell(addition_loc_layer))
                addition_cls_layer_list.append(nn.SequentialCell(addition_cls_layer))
            self.addition_layer_loc = nn.CellList(addition_loc_layer_list)
            self.addition_layer_cls = nn.CellList(addition_cls_layer_list)
        else:
            convs = [
                _conv2d(out_channels, out_channels, 3, 1) for x in range(num_addition_layers)
            ]
            addition_layer_list = []
            for _ in range(num_features):
                addition_layers = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, convs[x]) for x in range(num_addition_layers)
                ]
                addition_layer_list.append(nn.SequentialCell(addition_layers))
            self.addition_layer = nn.CellList(addition_layer_list)

        loc_layers = [_conv2d(out_channels, 4 * num_default,
                              kernel_size=3, stride=1, pad_mod='same')]
        cls_layers = [_conv2d(out_channels, num_classes * num_default,
                              kernel_size=3, stride=1, pad_mod='same')]

        self.loc_layers = nn.SequentialCell(loc_layers)
        self.cls_layers = nn.SequentialCell(cls_layers)
        self.flatten_concat = FlattenConcat(config)

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        num_heads = len(inputs)
        for i in range(num_heads):
            if self.loc_cls_shared_addition:
                features = self.addition_layer[i](inputs[i])
                loc_outputs += (self.loc_layers(features),)
                cls_outputs += (self.cls_layers(features),)
            else:
                features = self.addition_layer_loc[i](inputs[i])
                loc_outputs += (self.loc_layers(features),)
                features = self.addition_layer_cls[i](inputs[i])
                cls_outputs += (self.cls_layers(features),)
        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


# ==============================================================================
# MOBILENETV2 BACKBONE (CRITICAL FIX #1: Backbone-only feature extractor)
# ==============================================================================
class MobileNetV2Feature(nn.Cell):
    """
    MobileNetV2 backbone for SSD300 with proper feature extraction points.
    Outputs features at resolutions required by SSD300: 19x19 and 10x10
    """
    def __init__(self, width_mult=1.0, round_nearest=8):
        super(MobileNetV2Feature, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            # t, c, n, s (expand_ratio, output_channels, repeats, stride)
            [1, 16, 1, 1],
            [6, 24, 2, 2],   # Layer 1-2: 150x150 → 75x75
            [6, 32, 3, 2],   # Layer 3-5: 75x75 → 38x38
            [6, 64, 4, 2],   # Layer 6-9: 38x38 → 19x19 (LAYER 9 = 19x19 feature)
            [6, 96, 3, 1],   # Layer 10-12: 19x19 → 19x19
            [6, 160, 3, 2],  # Layer 13-15: 19x19 → 10x10 (LAYER 15 = 10x10 feature)
            [6, 320, 1, 1],  # Layer 16: 10x10 → 10x10
        ]

        # Building first layer (input 300x300 → 150x150)
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        
        # Building inverted residual blocks
        layer_index = 0
        self.feature_19x19_index = None  # Will be set at layer 9 (19x19 output)
        self.feature_10x10_index = None  # Will be set at layer 15 (10x10 output)
        
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                layer_index += 1
                
                # Mark feature extraction points
                if layer_index == 9:  # 19x19 feature map
                    self.feature_19x19_index = len(features) - 1
                elif layer_index == 15:  # 10x10 feature map
                    self.feature_10x10_index = len(features) - 1
        
        # Building last layer (1x1 conv to 1280 channels)
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        
        # Split features at extraction points
        self.features_19x19 = nn.SequentialCell(features[:self.feature_19x19_index + 1])  # Output: 19x19
        self.features_10x10 = nn.SequentialCell(features[self.feature_19x19_index + 1:self.feature_10x10_index + 1])  # 19x19 → 10x10
        self.features_tail = nn.SequentialCell(features[self.feature_10x10_index + 1:])  # 10x10 → 10x10 (1280 ch)
        
        # CRITICAL FIX #2: Channel projection layers to match SSD config expectations
        # MobileNetV2 native channels: 96@19x19, 320@10x10 → SSD expects 576@19x19, 1280@10x10
        self.proj_19x19 = nn.Conv2d(96, 576, kernel_size=1, has_bias=False)  # 96 → 576
        self.proj_10x10 = nn.Conv2d(320, 1280, kernel_size=1, has_bias=False)  # 320 → 1280
        self.bn_19x19 = _bn(576)
        self.bn_10x10 = _bn(1280)
        self.relu = nn.ReLU6()

    def construct(self, x):
        # Extract 19x19 feature map (layer 9 output)
        x_19x19 = self.features_19x19(x)  # Shape: [B, 96, 19, 19]
        
        # Project channels to match SSD expectations (96 → 576)
        feat_19x19 = self.relu(self.bn_19x19(self.proj_19x19(x_19x19)))  # [B, 576, 19, 19]
        
        # Continue to 10x10 feature map
        x_10x10 = self.features_10x10(x_19x19)  # Shape: [B, 320, 10, 10]
        
        # Project channels (320 → 1280)
        feat_10x10 = self.relu(self.bn_10x10(self.proj_10x10(x_10x10)))  # [B, 1280, 10, 10]
        
        # Final tail layers (optional enhancement)
        x_tail = self.features_tail(x_10x10)  # [B, 1280, 10, 10]
        
        return feat_19x19, feat_10x10


# ==============================================================================
# SSD300 WITH MOBILENETV2 BACKBONE (Proper integration)
# ==============================================================================
class SSD300MobileNetV2(nn.Cell):
    """
    SSD300 Network with MobileNetV2 backbone.
    Properly integrates MobileNetV2 features with SSD detection heads.
    """
    def __init__(self, backbone, config, is_training=True):
        super(SSD300MobileNetV2, self).__init__()
        self.backbone = backbone
        self.is_training = is_training
        
        # SSD extra layers (after backbone features)
        # Input channels: [576 (19x19), 1280 (10x10), ...] from config.extras_in_channels
        in_channels = config.extras_in_channels
        out_channels = config.extras_out_channels
        ratios = config.extras_ratio
        strides = config.extras_strides
        
        # Create residual layers for extra feature maps (5x5, 3x3, 2x2, 1x1)
        residual_list = []
        # Start from index 2 because first two features come from backbone (19x19, 10x10)
        for i in range(2, len(in_channels)):
            residual = InvertedResidual(
                in_channels[i], 
                out_channels[i], 
                stride=strides[i],
                expand_ratio=ratios[i], 
                last_relu=True
            )
            residual_list.append(residual)
        self.multi_residual = nn.CellList(residual_list)
        
        # Detection heads
        self.multi_box = MultiBox(config)
        
        # Inference activation
        if not is_training:
            self.activation = ops.Sigmoid()

    def construct(self, x):
        # Get backbone features: 19x19 and 10x10
        feat_19x19, feat_10x10 = self.backbone(x)
        
        # Build feature pyramid for SSD heads
        multi_feature = (feat_19x19, feat_10x10)
        feature = feat_10x10
        
        # Generate extra feature maps (5x5, 3x3, 2x2, 1x1)
        for residual in self.multi_residual:
            feature = residual(feature)
            multi_feature += (feature,)
        
        # Apply detection heads
        pred_loc, pred_label = self.multi_box(multi_feature)
        
        # Activation for inference mode
        if not self.is_training:
            pred_label = self.activation(pred_label)
        
        # Ensure float32 output
        pred_loc = ops.cast(pred_loc, ms.float32)
        pred_label = ops.cast(pred_label, ms.float32)
        return pred_loc, pred_label


# ==============================================================================
# EXISTING ARCHITECTURES (Preserved for backward compatibility)
# ==============================================================================
class SSD300(nn.Cell):
    """
    SSD300 Network. Default backbone is VGG16.
    """
    def __init__(self, backbone, config, is_training=True):
        super(SSD300, self).__init__()
        self.backbone = backbone
        self.is_training = is_training
        
        in_channels = config.extras_in_channels
        out_channels = config.extras_out_channels
        ratios = config.extras_ratio
        strides = config.extras_strides
        
        residual_list = []
        for i in range(2, len(in_channels)):
            residual = InvertedResidual(in_channels[i], out_channels[i], stride=strides[i],
                                        expand_ratio=ratios[i], last_relu=True)
            residual_list.append(residual)
        self.multi_residual = nn.CellList(residual_list)
        self.multi_box = MultiBox(config)
        
        if not is_training:
            self.activation = ops.Sigmoid()

    def construct(self, x):
        # VGG16 backbone returns (block4_3, block7)
        layer_out_13, output = self.backbone(x)
        multi_feature = (layer_out_13, output)
        feature = output
        
        for residual in self.multi_residual:
            feature = residual(feature)
            multi_feature += (feature,)
        
        pred_loc, pred_label = self.multi_box(multi_feature)
        if not self.is_training:
            pred_label = self.activation(pred_label)
        
        pred_loc = ops.cast(pred_loc, ms.float32)
        pred_label = ops.cast(pred_label, ms.float32)
        return pred_loc, pred_label


class SsdMobilenetV1Fpn(nn.Cell):
    def __init__(self, config):
        super(SsdMobilenetV1Fpn, self).__init__()
        self.multi_box = WeightSharedMultiBox(config)
        self.activation = ops.Sigmoid()
        self.feature_extractor = mobilenet_v1_fpn(config)

    def construct(self, x):
        features = self.feature_extractor(x)
        pred_loc, pred_label = self.multi_box(features)
        if not self.training:
            pred_label = self.activation(pred_label)
        pred_loc = ops.cast(pred_loc, ms.float32)
        pred_label = ops.cast(pred_label, ms.float32)
        return pred_loc, pred_label


class SsdMobilenetV1Feature(nn.Cell):
    def __init__(self, config, is_training=True):
        super(SsdMobilenetV1Feature, self).__init__()
        self.feature_extractor = mobilenet_v1_Feature(config)
        in_channels = config.extras_in_channels
        out_channels = config.extras_out_channels
        strides = config.extras_strides
        residual_list = []
        for i in range(2, len(in_channels)):
            residual = ConvBNReLU(in_channels[i], out_channels[i], stride=strides[i])
            residual_list.append(residual)
        self.multi_residual = nn.CellList(residual_list)
        self.multi_box = MultiBox(config)
        self.is_training = is_training
        if not is_training:
            self.activation = ops.Sigmoid()

    def construct(self, x):
        feature, output = self.feature_extractor(x)
        multi_feature = (feature, output)
        feature = output
        for residual in self.multi_residual:
            feature = residual(feature)
            multi_feature += (feature,)
        pred_loc, pred_label = self.multi_box(multi_feature)
        if not self.training:
            pred_label = self.activation(pred_label)
        pred_loc = ops.cast(pred_loc, ms.float32)
        pred_label = ops.cast(pred_label, ms.float32)
        return pred_loc, pred_label


class SsdResNet50Fpn(nn.Cell):
    def __init__(self, config):
        super(SsdResNet50Fpn, self).__init__()
        self.multi_box = WeightSharedMultiBox(config)
        self.activation = ops.Sigmoid()
        self.feature_extractor = resnet50_fpn()

    def construct(self, x):
        features = self.feature_extractor(x)
        pred_loc, pred_label = self.multi_box(features)
        if not self.training:
            pred_label = self.activation(pred_label)
        pred_loc = ops.cast(pred_loc, ms.float32)
        pred_label = ops.cast(pred_label, ms.float32)
        return pred_loc, pred_label


class SigmoidFocalClassificationLoss(nn.Cell):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.sigmiod_cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.sigmoid = ops.Sigmoid()
        self.pow = ops.Pow()
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.gamma = gamma
        self.alpha = alpha

    def construct(self, logits, label):
        label = self.onehot(label, ops.shape(logits)[-1], self.on_value, self.off_value)
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)
        sigmoid = self.sigmoid(logits)
        label = ops.cast(label, ms.float32)
        p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
        modulating_factor = self.pow(1 - p_t, self.gamma)
        alpha_weight_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
        return focal_loss


class SSDWithLossCell(nn.Cell):
    def __init__(self, network, config):
        super(SSDWithLossCell, self).__init__()
        self.network = network
        self.less = ops.Less()
        self.tile = ops.Tile()
        self.reduce_sum = ops.ReduceSum()
        self.expand_dims = ops.ExpandDims()
        self.class_loss = SigmoidFocalClassificationLoss(config.gamma, config.alpha)
        self.loc_loss = nn.SmoothL1Loss()

    def construct(self, x, gt_loc, gt_label, num_matched_boxes):
        pred_loc, pred_label = self.network(x)
        mask = ops.cast(self.less(0, gt_label), ms.float32)
        num_matched_boxes = self.reduce_sum(ops.cast(num_matched_boxes, ms.float32))

        # Localization Loss
        mask_loc = self.tile(self.expand_dims(mask, -1), (1, 1, 4))
        smooth_l1 = self.loc_loss(pred_loc, gt_loc) * mask_loc
        loss_loc = self.reduce_sum(self.reduce_sum(smooth_l1, -1), -1)

        # Classification Loss
        loss_cls = self.class_loss(pred_label, gt_label)
        loss_cls = self.reduce_sum(loss_cls, (1, 2))

        return self.reduce_sum((loss_cls + loss_loc) / num_matched_boxes)


grad_scale = ops.MultitypeFuncGraph("grad_scale")
@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.Reciprocal()(scale)


class TrainingWrapper(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0, use_global_norm=False):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.use_global_norm = use_global_norm
        self.parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = ms.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = ms.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        self.hyper_map = ops.HyperMap()

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        if self.use_global_norm:
            grads = self.hyper_map(ops.partial(grad_scale, ops.scalar_to_tensor(self.sens)), grads)
            grads = ops.clip_by_global_norm(grads)
        self.optimizer(grads)
        return loss


class SsdInferWithDecoder(nn.Cell):
    def __init__(self, network, default_boxes, config):
        super(SsdInferWithDecoder, self).__init__()
        self.network = network
        self.default_boxes = default_boxes
        self.prior_scaling_xy = config.prior_scaling[0]
        self.prior_scaling_wh = config.prior_scaling[1]

    def construct(self, x):
        pred_loc, pred_label = self.network(x)

        default_bbox_xy = self.default_boxes[..., :2]
        default_bbox_wh = self.default_boxes[..., 2:]
        pred_xy = pred_loc[..., :2] * self.prior_scaling_xy * default_bbox_wh + default_bbox_xy
        pred_wh = ops.Exp()(pred_loc[..., 2:] * self.prior_scaling_wh) * default_bbox_wh

        pred_xy_0 = pred_xy - pred_wh / 2.0
        pred_xy_1 = pred_xy + pred_wh / 2.0
        pred_xy = ops.Concat(-1)((pred_xy_0, pred_xy_1))
        pred_xy = ops.Maximum()(pred_xy, 0)
        pred_xy = ops.Minimum()(pred_xy, 1)
        return pred_xy, pred_label


# ==============================================================================
# FACTORY FUNCTIONS (CRITICAL FIX: Return proper backbone instances)
# ==============================================================================
def ssd_mobilenet_v1_fpn(**kwargs):
    return SsdMobilenetV1Fpn(**kwargs)

def ssd_mobilenet_v1(**kwargs):
    return SsdMobilenetV1Feature(**kwargs)

def ssd_resnet50_fpn(**kwargs):
    return SsdResNet50Fpn(**kwargs)

def ssd_mobilenet_v2():
    """
    Factory function returning MobileNetV2 backbone ONLY (not full SSD network).
    Compatible with SSD300MobileNetV2 wrapper.
    """
    return MobileNetV2Feature()

def ssd_vgg16(**kwargs):
    return SSD300VGG16(**kwargs)


# ==============================================================================
# VGG16 SSD300 (Preserved for backward compatibility)
# ==============================================================================
class SSD300VGG16(nn.Cell):
    def __init__(self, config):
        super(SSD300VGG16, self).__init__()
        self.backbone = vgg16()
        self.b6_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6, pad_mode='pad')
        self.b6_2 = nn.Dropout(0.5)
        self.b7_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.b7_2 = nn.Dropout(0.5)
        self.b8_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, padding=1, pad_mode='pad')
        self.b8_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, pad_mode='valid')
        self.b9_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=1, pad_mode='pad')
        self.b9_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, pad_mode='valid')
        self.b10_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.b10_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, pad_mode='valid')
        self.b11_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.b11_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, pad_mode='valid')
        self.multi_box = MultiBox(config)
        if not self.training:
            self.activation = ops.Sigmoid()

    def construct(self, x):
        block4, x = self.backbone(x)
        x = self.b6_1(x)
        x = self.b6_2(x)
        x = self.b7_1(x)
        x = self.b7_2(x)
        block7 = x
        x = self.b8_1(x)
        x = self.b8_2(x)
        block8 = x
        x = self.b9_1(x)
        x = self.b9_2(x)
        block9 = x
        x = self.b10_1(x)
        x = self.b10_2(x)
        block10 = x
        x = self.b11_1(x)
        x = self.b11_2(x)
        block11 = x
        multi_feature = (block4, block7, block8, block9, block10, block11)
        pred_loc, pred_label = self.multi_box(multi_feature)
        if not self.training:
            pred_label = self.activation(pred_label)
        pred_loc = ops.cast(pred_loc, ms.float32)
        pred_label = ops.cast(pred_label, ms.float32)
        return pred_loc, pred_label

import math
import numpy as np


def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(float): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_end + \
                 (lr_max - lr_end) * \
                 (1. + math.cos(math.pi * (i - warmup_steps) / (total_steps - warmup_steps))) / 2.
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def get_lr_cosine(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    """
    Cosine decay learning rate schedule with warmup
    Essential for MobileNetV2 stability
    """
    import numpy as np
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            # Linear warmup
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            # Cosine decay
            decay_steps = total_steps - warmup_steps
            decay_progress = (i - warmup_steps) / decay_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_progress))
            lr = (lr_max - lr_end) * cosine_decay + lr_end
        lr_each_step.append(lr)
    
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step[global_step:]

# Update get_lr() to dispatch based on config:
def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch, lr_decay_mode="cosine"):
    if lr_decay_mode == "cosine":
        return get_lr_cosine(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch)
    elif lr_decay_mode == "steps":
        return get_lr_steps(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch)
    else:
        raise ValueError(f"Unsupported lr_decay_mode: {lr_decay_mode}")

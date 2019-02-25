from torch.optim.lr_scheduler import _LRScheduler


class PolynomialDecayLR(_LRScheduler):
    """
    A scheduler which applies a polynomial decay to the learning rate
    with idea from https://www.tensorflow.org/api_docs/python/tf/train/polynomial_decay
    """

    def __init__(self, optimizer, max_decay_steps, last_decay_step=-1, power=2,  learning_rate_end=1e-6):
        self.optimizer = optimizer
        self.max_decay_steps = max_decay_steps
        self.global_step = last_decay_step
        self.learning_rate_end = learning_rate_end
        self.power = power

        if last_decay_step == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_decay_step + 1)

    def get_lr(self):
        return [self.lr_decay_func(base_lr) for base_lr in self.base_lrs]

    def lr_decay_func(self, start_lr):
        self.global_step = min(self.global_step, self.max_decay_steps)
        new_lr = (start_lr - self.learning_rate_end) * ((1 - self.global_step / self.max_decay_steps) ** self.power) + self.learning_rate_end
        return new_lr

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.global_step + 1
        self.global_step = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

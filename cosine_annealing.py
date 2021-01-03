import math
from keras.callbacks import Callback
from keras import backend as K


class CosineAnnealingScheduler(Callback):
    """
    Cosine annealing scheduler.
    """

    def __init__(self, T_i, eta_max, eta_min, increaseTMAX, T_mult=2, verbose=1):
        """
        Args:
            increaseTMAX: Whether increase T_i or not; otherwise T_i will be fixed as provided.
            T_i: span.
            epoch_curr: How many epochs have been performed since last restart. {0,1,2,3,...etc.}
            epoch_run_i: Check point of last restart ended.
        """
        super(CosineAnnealingScheduler, self).__init__()
        self.T_i = T_i
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.increaseTMAX = increaseTMAX
        self.T_mult = T_mult
        self.verbose = verbose
        self.epoch_run_i = 0
        self.epoch_curr = 0

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        self.epoch_curr = epoch - self.epoch_run_i
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.epoch_curr / self.T_i)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %04d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_curr == self.T_i - 1:
            self.epoch_run_i = epoch + 1
            if self.increaseTMAX:
                self.T_i = self.T_i * self.T_mult
        #  print("Epoch current: {}\nEpoch_run_i: {}\n, T_i: {}".format(self.epoch_curr, self.epoch_run_i, self.T_i))
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

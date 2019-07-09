from keras.callbacks import Callback
import keras.backend as K

class LrReducer(Callback):
    def __init__(self, monitor=None, patience=None, reduce_rate=None, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.wait = 0
        self.best_score = 1000.
        self.reduce_rate = reduce_rate
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get(self.monitor)
        if current_score < self.best_score:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print('Epoch %05d: current best val_loss is: %.5f\n' % (epoch+1, current_score))
        else:
            if self.wait >= self.patience:
                lr = K.get_value(self.model.optimizer.lr)
                lr_new = lr*self.reduce_rate
                K.set_value(self.model.optimizer.lr, lr_new)
                print('Epoch %05d: LrReducer reduce learning rate to %s.\n' % (epoch + 1, lr_new))
                self.wait = 0
            else:
                self.wait += 1
                    
                    
                    
#             if self.wait >= self.patience:
#                 self.current_reduce_nb += 1
#                 if self.current_reduce_nb <= self.early_stopping_num:
#                     lr = K.get_value(self.model.optimizer.lr)
#                     lr_new = lr*self.reduce_rate
#                     K.set_value(self.model.optimizer.lr, lr_new)
#                     print('Epoch %05d: LrReducer reduce learning rate to %s.\n' % (epoch + 1, lr_new))
#                 else:
#                     if self.verbose > 0:
#                         print("Epoch %05d: early stopping.\n" % (epoch))
#                     self.model.stop_training = True
#             self.wait += 1
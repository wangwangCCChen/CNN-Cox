from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from lifelines.utils import concordance_index

#Callback
class MyCallback(ModelCheckpoint):
    def __init__(self, filepath, data, real_save=True, patience=20):
        super(MyCallback, self).__init__(filepath, save_weights_only=True)
        self.patience = patience

        self.x_trn, self.c_trn, self.s_trn, self.x_dev, self.c_dev, self.s_dev = data

        self.cindex_dev = 0
        self.cindex_best_epoch = 0
        self.real_save = real_save
        self.filepath_template = self.filepath+'-%s'
        self.max_epoch = 1000

    def print_status(self):
        print('\n=========================== [Best cindex (epoch = %d)] cindex=%f =================================='
              % (self.cindex_best_epoch, self.cindex_dev))


    def on_train_end(self, logs=None):
        print('[Best:on_train_end]')
        self.print_status()

    def on_epoch_end(self, epoch, logs=None):
        pred_dev = -np.exp(self.model.predict(self.x_dev, batch_size=1, verbose=0))
        cindex_dev = concordance_index(self.s_dev, pred_dev, self.c_dev)

        if self.cindex_dev < cindex_dev:
            self.cindex_dev = cindex_dev
            self.cindex_best_epoch = epoch
            if self.real_save is True:
                if self.save_weights_only:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)

        else:
            if epoch - self.cindex_best_epoch > self.patience:
                self.model.stop_training = True
                print("Early stopping at %d" % epoch)

        if epoch > self.max_epoch:
                self.model.stop_training = True
                print("Stopping at max epoch %d" % epoch)
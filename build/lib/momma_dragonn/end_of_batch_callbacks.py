import keras.callbacks
import pdb 
class BatchQC(keras.callbacks.Callback):
    def on_batch_begin(self,batch,logs={}):
        return
    def on_batch_end(self,batch,logs={}):
        pdb.set_trace() 
        #predictions_train_batch=self.model.predict(
        #self.losses.append(logs.get('loss'))
        return 

from .core import AbstractBatchDataLoader, AbstractAtOnceDataLoader
import h5py
import numpy as np
from avutils import util


class MultimodalBatchDataLoader(AbstractBatchDataLoader):

    def __init__(self, path_to_hdf5, **kwargs):
        super(MultimodalBatchDataLoader, self).__init__(**kwargs)
        self.f = h5py.File(path_to_hdf5)
        self.X = f['/X']
        self.Y = f['/Y']
        assert len(self.X) == len(self.Y)
        self.num_items = len(self.X)
        self.start_index = 0

    def get_batch_generator(self):
        end_index = min(self.num_items, self.start_index+self.batch_size)
        X_batch = {}
        Y_batch = {}
        for input_mode in self.X:
            X_batch[input_mode] = self.X[input_mode]\
                                        [self.start_index:end_index] 
        for output_mode in self.Y:
            Y_batch[output_mode] = self.Y[output_mode]\
                                         [self.start_index:end_index]
        if (end_index==self.num_items):
            self.start_index = 0 
        return X_batch, Y_batch
        
 
class MultimodalAtOnceDataLoader(AbstractAtOnceDataLoader)

    def __init__(self, path_to_hdf5, **kwargs):
        super(MultimodalAtOnceDataLoader, self).__init__(**kwargs)
        self.f = h5py.File(path_to_hdf5)
        self.X = f['/X']
        self.Y = f['/X']

    def get_data(self):
        X = {}
        Y = {}
        for input_mode in self.X:
            X[input_mode] = np.array(self.X[input_mode])
        for output_mode in self.Y:
            Y[output_mode] = np.array(self.Y[output_mode])
        return util.enum(X=X, Y=Y)

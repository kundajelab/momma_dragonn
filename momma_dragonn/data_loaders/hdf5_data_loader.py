from .core import AbstractBatchDataLoader, AbstractAtOnceDataLoader
import h5py
import numpy as np
from avutils import util


class MultimodalBatchDataLoader(AbstractBatchDataLoader):

    def __init__(self, path_to_hdf5, num_to_load_for_eval, **kwargs):
        super(MultimodalBatchDataLoader, self).__init__(**kwargs)
        self.path_to_hdf5 = path_to_hdf5
        self.f = h5py.File(self.path_to_hdf5)
        self.X = f['/X']
        self.Y = f['/Y']
        assert len(self.X) == len(self.Y)
        self.num_items = len(self.X)
        self.start_index = 0
        self.num_to_load_for_eval = num_to_load_for_eval

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

    def get_data_for_eval(self):
        X = {}
        Y = {}
        eval_start_index = max(self.start_index-self.num_to_load_for_eval,0)
        eval_end_index = self.start_index
        for input_mode in self.X:
            #load the last self.num_to_load_for_eval
            X[input_mode] = self.X[input_mode]\
                                  [eval_start_index:eval_end_index]

        for output_mode in self.Y:
            Y[output_mode] = self.Y[output_mode]\
                                   [eval_start_index:eval_end_index]
        return util.enum(X=X,Y=Y)
             
    def get_jsonable_object(self):
        the_dict = super(MultimodalBatchDataLoader, self).get_jsonable_object() 
        the_dict['path_to_hdf5'] = self.path_to_hdf5
        the_dict['num_to_load_for_eval'] = self.num_to_load_for_eval
        return the_dict 

 
class MultimodalAtOnceDataLoader(AbstractAtOnceDataLoader)

    def __init__(self, path_to_hdf5, **kwargs):
        super(MultimodalAtOnceDataLoader, self).__init__(**kwargs)
        self.path_to_hdf5 = path_to_hdf5
        self.f = h5py.File(self.path_to_hdf5)
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

    def get_jsonable_object(self):
        the_dict = super(MultimodalAtOnceDataLoader, self)\
                   .get_jsonable_object()
        the_dict['path_to_hdf5'] = self.path_to_hdf5
        return the_dict

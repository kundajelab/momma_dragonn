from .core import AbstractBatchDataLoader, AbstractAtOnceDataLoader
import h5py
import numpy as np
from avutils import util


class MultimodalBatchDataLoader(AbstractBatchDataLoader):

    def __init__(self, path_to_hdf5, num_to_load_for_eval, 
                       bundle_x_and_y_in_generator, **kwargs):
        super(MultimodalBatchDataLoader, self).__init__(**kwargs)
        self.path_to_hdf5 = path_to_hdf5
        self.bundle_x_and_y_in_generator = bundle_x_and_y_in_generator
        self.f = h5py.File(self.path_to_hdf5)
        self.X = self.f['/X']
        self.Y = self.f['/Y']
        if ('weight' in self.f):
            self.weight = self.f['/weight']
        else:
            self.weight = {}
        self.input_modes = self.X.keys()
        print("Input modes",self.input_modes)
        self.output_modes = self.Y.keys()
        print("Output modes",self.output_modes)
        assert len(self.X) == len(self.Y)
        self.num_items = len(self.X[self.input_modes[0]])
        if (num_to_load_for_eval > self.num_items):
            print("num_to_load_for_eval is ",num_to_load_for_eval,
                  "but num_items is",self.num_items,"- reducing")
            num_to_load_for_eval = self.num_items
        self.start_index = 0
        self.num_to_load_for_eval = num_to_load_for_eval
             
    def get_jsonable_object(self):
        the_dict = super(MultimodalBatchDataLoader, self).get_jsonable_object() 
        the_dict['path_to_hdf5'] = self.path_to_hdf5
        the_dict['num_to_load_for_eval'] = self.num_to_load_for_eval
        the_dict['bundle_x_and_y_in_generator'] =\
            self.bundle_x_and_y_in_generator
        return the_dict 

    def get_batch_generator(self):
        while True:
            if (self.start_index==self.num_items):
                self.start_index = 0 
            end_index = min(self.num_items, self.start_index+self.batch_size)
            x_batch = {}
            y_batch = {}
            weight_batch = {}
            for input_mode in self.input_modes:
                x_batch[input_mode] = self.X[input_mode]\
                                            [self.start_index:end_index] 

            for output_mode in self.output_modes:
                y_batch[output_mode] = self.Y[output_mode]\
                                             [self.start_index:end_index]
            for output_mode in self.weight:
                weight_batch[output_mode] = self.weight[output_mode]\
                                                 [self.start_index+end_index]

            self.start_index = end_index

            if (self.bundle_x_and_y_in_generator):
                data_batch = {}
                data_batch.update(x_batch)
                data_batch.update(y_batch)
                yield (data_batch, weight_batch)
            else:
                yield (x_batch, y_batch, weight_batch)

    def get_data_for_eval(self):
        X = {}
        Y = {}
        #take the items immediately preceding the current start_index
        eval_start_index_1 = max(self.start_index-self.num_to_load_for_eval,0)
        eval_end_index_1 = self.start_index
        #any leftover taken from the end (presumably last seen)
        eval_start_index_2 = self.num_items-\
                             max(self.num_to_load_for_eval-self.start_index,0)
        eval_end_index_2 = self.num_items
        for input_mode in self.X:
            #load the last self.num_to_load_for_eval
            arr1 = self.X[input_mode][eval_start_index_1:eval_end_index_1]
            arr2 = self.X[input_mode][eval_start_index_2:eval_end_index_2]
            X[input_mode] = np.concatenate([arr1, arr2], axis=0)

        for output_mode in self.Y:
            arr1 = self.Y[output_mode][eval_start_index_1:eval_end_index_1]
            arr2 = self.Y[output_mode][eval_start_index_2:eval_end_index_2]
            Y[output_mode] = np.concatenate([arr1, arr2], axis=0)
        return util.enum(X=X,Y=Y)

 
class MultimodalAtOnceDataLoader(AbstractAtOnceDataLoader):

    def __init__(self, path_to_hdf5, max_to_load=None, **kwargs):
        super(MultimodalAtOnceDataLoader, self).__init__(**kwargs)
        self.path_to_hdf5 = path_to_hdf5
        self.max_to_load = max_to_load
        self.f = h5py.File(self.path_to_hdf5)
        self.X = self.f['/X']
        self.Y = self.f['/Y']

    def get_data(self):
        X = {}
        Y = {}
        for input_mode in self.X:
            if (self.max_to_load is None):
                X[input_mode] = np.array(self.X[input_mode])
            else:
                X[input_mode] = np.array(self.X[input_mode][:self.max_to_load])
        for output_mode in self.Y:
            if (self.max_to_load is None):
                Y[output_mode] = np.array(self.Y[output_mode])
            else:
                Y[output_mode] = np.array(self.Y[output_mode]
                                                [:self.max_to_load])
        return util.enum(X=X, Y=Y)

    def get_jsonable_object(self):
        the_dict = super(MultimodalAtOnceDataLoader, self)\
                   .get_jsonable_object()
        the_dict['path_to_hdf5'] = self.path_to_hdf5
        the_dict['max_to_load'] = self.max_to_load
        return the_dict

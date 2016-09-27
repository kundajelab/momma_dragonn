from collections import OrderedDict
import numpy as np
from avutils import util


class AbstractDataLoader(object):

    def get_jsonable_object(self):
        return OrderedDict()

    def get_callbacks(self):
        return [] #TODO: implement callback functionality in a trainer
        

class AbstractBatchDataLoader(AbstractDataLoader):

    def __init__(self, batch_size, num_to_load_for_eval):
        self.batch_size = batch_size
        self.num_to_load_for_eval = num_to_load_for_eval

    def get_jsonable_object(self):
        the_dict = super(AbstractBatchDataLoader, self).get_jsonable_object()
        the_dict['batch_size'] = self.batch_size
        the_dict['num_to_load_for_eval'] = self.num_to_load_for_eval
        return the_dict

    def get_batch_generator(self):
        #produces the generator for batches 
        raise NotImplementedError()

    def get_data_for_eval(self):
        #produce a hunk of data to run evaluation on
        raise NotImplementedError()


class BatchDataLoader_XYDictAPI(AbstractBatchDataLoader):

    def __init__(self, X, Y, weight, num_to_load_for_eval,
                       bundle_x_and_y_in_generator, **kwargs):
        super(BatchDataLoader_XYDictAPI, self).__init__(**kwargs)
        self.X = X
        self.Y = Y
        self.weight = weight

        self.bundle_x_and_y_in_generator = bundle_x_and_y_in_generator

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
        self.num_to_load_for_eval = num_to_load_for_eval

        self.start_index = 0
 
    def get_jsonable_object(self):
        the_dict = super(BatchDataLoader_XYDictAPI, self)\
                   .get_jsonable_object() 
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


class AbstractAtOnceDataLoader(AbstractDataLoader):

    def get_data(self):
        raise NotImplementedError()


class AtOnceDataLoader_XYDictAPI(AbstractAtOnceDataLoader):

    def __init__(self, X, Y, max_to_load=None, **kwargs):
        super(AtOnceDataLoader_XYDictAPI, self).__init__(**kwargs)
        self.X = X
        self.Y = Y
        self.max_to_load = max_to_load

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
        the_dict['max_to_load'] = self.max_to_load
        return the_dict

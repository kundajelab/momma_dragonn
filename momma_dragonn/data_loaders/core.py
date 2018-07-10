from __future__ import print_function
from collections import OrderedDict
import numpy as np
from avutils import util


class AbstractDataLoader(object):

    def get_jsonable_object(self):
        return OrderedDict()

    def get_callbacks(self):
        return [] #TODO: implement callback functionality in a trainer
        

class AbstractBatchDataLoader(AbstractDataLoader):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_jsonable_object(self):
        the_dict = super(AbstractBatchDataLoader, self).get_jsonable_object()
        the_dict['batch_size'] = self.batch_size
        return the_dict

    def get_batch_generator(self):
        #produces the generator for batches 
        raise NotImplementedError()

    def get_data_for_eval(self):
        #produce a hunk of data to run evaluation on
        raise NotImplementedError()

    def get_data(self):
        #returns an object with pointers to all the X and Y
        #data at once; only needed for validation set, which
        #as of the time of writing
        #should be small enough to fit in memory
        raise NotImplementedError()


class BatchDataLoader_XYDictAPI(AbstractBatchDataLoader):

    def __init__(self, X, Y, weight, bundle_x_and_y_in_generator,
                       num_to_load_for_eval,
                       strip_enclosing_dictionary=False,
                       rc_augment=False,
                       **kwargs):
        super(BatchDataLoader_XYDictAPI, self).__init__(**kwargs)
        self.X = X
        self.Y = Y
        self.weight = weight
        self.rc_augment = rc_augment
        if (self.rc_augment):
            assert len(self.X.keys()) == 1
            assert len(self.X[self.X.keys()[0]].shape)==3
        self.strip_enclosing_dictionary = strip_enclosing_dictionary
        if (strip_enclosing_dictionary): #for sequential models
            assert len(X.keys())==1
            assert len(Y.keys())==1

        self.bundle_x_and_y_in_generator = bundle_x_and_y_in_generator

        self.input_modes = list(self.X.keys())
        print("Input modes",self.input_modes)
        self.output_modes = list(self.Y.keys())
        print("Output modes",self.output_modes)

        assert (len(self.X[list(self.X.keys())[0]])
                == len(self.Y[list(self.Y.keys())[0]])), (str(len(self.X))
                                                    +"\t"+str(len(self.Y)))

        self.num_items = len(self.X[self.input_modes[0]])
        if (num_to_load_for_eval is None or
            num_to_load_for_eval > self.num_items):
            if (num_to_load_for_eval is not None):
                print("num_to_load_for_eval is ",num_to_load_for_eval,
                      "but num_items is",self.num_items,"- fixing")
            num_to_load_for_eval = self.num_items
        self.num_to_load_for_eval = num_to_load_for_eval

        self.start_index = 0
 
    def get_jsonable_object(self):
        the_dict = super(BatchDataLoader_XYDictAPI, self)\
                   .get_jsonable_object() 
        the_dict['num_to_load_for_eval'] = self.num_to_load_for_eval
        the_dict['bundle_x_and_y_in_generator'] =\
            self.bundle_x_and_y_in_generator
        the_dict['strip_enclosing_dictionary'] =\
            self.strip_enclosing_dictionary 
        the_dict['rc_augment'] = self.rc_augment
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
                the_arr = self.X[input_mode][self.start_index:end_index] 
                if (self.rc_augment):
                    the_arr = np.concatenate([the_arr, the_arr[:,::-1,::-1]],
                                             axis=0)
                if (self.strip_enclosing_dictionary):
                    x_batch = the_arr
                else:
                    x_batch[input_mode] = the_arr

            for output_mode in self.output_modes:
                the_arr = self.Y[output_mode][self.start_index:end_index]
                if (self.rc_augment):
                    the_arr = np.concatenate([the_arr, the_arr],
                                             axis=0)
                if (self.strip_enclosing_dictionary):
                    y_batch = the_arr
                else:
                    y_batch[output_mode] = the_arr 

            for output_mode in self.weight:
                the_arr = self.weight[output_mode][self.start_index:end_index]
                if (self.rc_augment):
                    the_arr = np.concatenate([the_arr, the_arr],
                                             axis=0)
                if (self.strip_enclosing_dictionary):
                    weight_batch = the_arr
                else:
                    weight_batch[output_mode] = the_arr
                

            self.start_index = end_index

            if (self.bundle_x_and_y_in_generator):
                data_batch = {}
                data_batch.update(x_batch)
                data_batch.update(y_batch)
                yield (data_batch, weight_batch)
            else:
                if (len(weight_batch)==0):
                    yield(x_batch, y_batch)
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
            the_arr = np.concatenate([arr1, arr2], axis=0)
            if (self.strip_enclosing_dictionary):
                X = the_arr
            else:
                X[input_mode] = the_arr

        for output_mode in self.Y:
            arr1 = self.Y[output_mode][eval_start_index_1:eval_end_index_1]
            arr2 = self.Y[output_mode][eval_start_index_2:eval_end_index_2]
            the_arr = np.concatenate([arr1, arr2], axis=0)
            if (self.strip_enclosing_dictionary):
                Y = the_arr 
            else:
                Y[output_mode] = the_arr
        return util.enum(X=X,Y=Y)

    def get_data(self):
        if (self.strip_enclosing_dictionary):
            X = self.X[list(self.X.keys())[0]] 
            Y = self.Y[list(self.Y.keys())[0]]
            return util.enum(X=X, Y=Y)
        else: 
            return util.enum(X=self.X, Y=self.Y)


class AtOnceDataLoader_XYDictAPI(BatchDataLoader_XYDictAPI):

    def __init__(self, X_full, Y_full,
                       strip_enclosing_dictionary=False,
                       max_to_load=None,
                       max_to_load_is_random=False,
                       seed=1,
                       #arguments below are only relevant if
                       #want to use in batches as well; listing them
                       #out here to indicate their defaults
                       num_to_load_for_eval=None,
                       bundle_x_and_y_in_generator=None,
                       batch_size=None, 
                       rc_augment=False,
                       **kwargs):

        self.strip_enclosing_dictionary = strip_enclosing_dictionary
        self.max_to_load = max_to_load
        self.max_to_load_is_random = max_to_load_is_random
        self.seed = seed
	
        if (strip_enclosing_dictionary): #for sequential models
            assert len(X_full.keys())==1
            assert len(Y_full.keys())==1

        X = {}
        Y = {}
        for input_mode in X_full:
            if (self.max_to_load is None):
                the_arr = np.array(X_full[input_mode])
            else:
                if (max_to_load_is_random == False):
                    the_arr = np.array(X_full[input_mode][:self.max_to_load])
                else:
                    np.random.seed(seed) # ensure that the validation set is reproducibly chosen
                    the_arr = np.random.choice(X_full[input_mode], size = self.max_to_load, replace = False)
            X[input_mode] = the_arr
        for output_mode in Y_full:
            if (self.max_to_load is None):
                the_arr = np.array(Y_full[output_mode])
            else:
                if (max_to_load_is_random == False):
                    the_arr = np.array(Y_full[output_mode][:self.max_to_load])
                else:
                    the_arr = np.random.choice(Y_full[output_mode], size = self.max_to_load, replace = False)
            Y[output_mode] = the_arr

        super(AtOnceDataLoader_XYDictAPI, self).__init__(
            X=X, Y=Y,
            weight={},
            num_to_load_for_eval=num_to_load_for_eval,
            bundle_x_and_y_in_generator=bundle_x_and_y_in_generator,
            batch_size=batch_size,
            strip_enclosing_dictionary=strip_enclosing_dictionary,
            rc_augment=rc_augment,
            **kwargs)

    def get_jsonable_object(self):
        the_dict = super(AtOnceDataLoader_XYDictAPI, self)\
                   .get_jsonable_object()
        the_dict['max_to_load'] = self.max_to_load
        the_dict['max_to_load_is_random'] = self.max_to_load_is_random
        the_dict['seed'] = self.seed
        return the_dict

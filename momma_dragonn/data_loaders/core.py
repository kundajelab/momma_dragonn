from collections import OrderedDict
import numpy as np
from avutils import util
import pdb

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

#updated to randomize batch selection s.t. all data points get examined in a single epoch of training
class BatchDataLoader_XYDictAPI(AbstractBatchDataLoader):

    def __init__(self, X, Y, weight, bundle_x_and_y_in_generator,use_weights=True,
                       num_to_load_for_eval =None,percent_to_load_for_eval=100, **kwargs):
        super(BatchDataLoader_XYDictAPI, self).__init__(**kwargs)
        self.X = X
        self.Y = Y
        self.weight = weight
        self.use_weights=use_weights 
        self.bundle_x_and_y_in_generator = bundle_x_and_y_in_generator

        self.input_modes = self.X.keys()
        print("Input modes",self.input_modes)
        self.output_modes = self.Y.keys()
        print("Output modes",self.output_modes)
        
        #assertion needs to be made for each task.
        for x_key in self.X.keys():
            for y_key in self.Y.keys(): 
                assert len(self.X[x_key]) == len(self.Y[y_key])

        self.num_items = len(self.X[self.input_modes[0]])

        if (num_to_load_for_eval > self.num_items):
            print("num_to_load_for_eval is ",num_to_load_for_eval,"but num_items is",self.num_items,"- fixing")
            num_to_load_for_eval = self.num_items
        if (num_to_load_for_eval!=None) and (percent_to_load_for_eval!=None):
            print("You have specified both number_to_load_for_eval and percent_to_load_for_eval. The algorithm will default to the percent_to_load_for_eval")
        if (percent_to_load_for_eval!=None):
            self.num_to_load_for_eval=int((float(percent_to_load_for_eval)/100)*self.num_items)
        elif (num_to_load_for_Eval!=None):
            self.num_to_load_for_eval = num_to_load_for_eval
        else:
            self.num_to_load_for_eval=self.num_items 
        self.permuted_batch_start,self.permuted_batch_end=self.get_batch_permutation_order()
        
    def get_jsonable_object(self):
        the_dict = super(BatchDataLoader_XYDictAPI, self)\
                   .get_jsonable_object() 
        the_dict['num_to_load_for_eval'] = self.num_to_load_for_eval
        the_dict['bundle_x_and_y_in_generator'] =\
            self.bundle_x_and_y_in_generator
        return the_dict
    
    #generate a permutation of batches to use
    def get_batch_permutation_order(self):
        #figure out what the offset of the first batch from 0 is
        start_index=np.random.randint(0,self.batch_size)
        batch_number=0
        ordered_batches=OrderedDict()
        while start_index < self.num_items:
            end_index=start_index+self.batch_size
            #handle the edge case when your end_index runs off the sequence!
            if end_index >=self.num_items:
                #shift left
                print("shifting!!") 
                end_index=self.num_items
                start_index=end_index-self.batch_size 
            ordered_batches[batch_number]=[start_index,end_index]
            batch_number+=1
            start_index=start_index+self.batch_size
            if start_index%1000==0:
                print str(start_index) 
        #permute the batch indices
        print("finished permutation") 
        permuted_indices=np.random.permutation(batch_number)
        #return start & end indices for the permuted batches
        permuted_batch_start=[]
        permuted_batch_end=[] 
        for i in permuted_indices:
            permuted_batch_start.append(ordered_batches[i][0])
            permuted_batch_end.append(ordered_batches[i][1])
        return permuted_batch_start, permuted_batch_end
    

    #this is for the training step, we cycle through all examples 
    def get_batch_generator(self):   
        num_generated=0
        num_expected=self.num_items
        self.permutation_index=0 
        while True:            
            if (num_generated>=num_expected) or (self.permutation_index >= len(self.permuted_batch_start)):
                self.permuted_batch_start,self.permuted_batch_end=self.get_batch_permutation_order()
                self.permutation_index=0
            x_batch = {}
            y_batch = {}
            weight_batch = {}
            for input_mode in self.input_modes:
                x_batch[input_mode] = np.asarray(self.X[input_mode]\
                                            [self.permuted_batch_start[self.permutation_index]:self.permuted_batch_end[self.permutation_index]])

            for output_mode in self.output_modes:
                y_batch[output_mode] = np.asarray(self.Y[output_mode]\
                                             [self.permuted_batch_start[self.permutation_index]:self.permuted_batch_end[self.permutation_index]])
            for output_mode in self.weight:
                weight_batch[output_mode] = np.asarray(self.weight[output_mode]\
                                            [self.permuted_batch_start[self.permutation_index]:self.permuted_batch_end[self.permutation_index]])
            self.permutation_index+=1
            if (self.bundle_x_and_y_in_generator):
                #data_batch = tuple([x_batch,y_batch])
                data_batch={} 
                data_batch.update(x_batch)
                data_batch.update(y_batch)
                if self.use_weights:
                    yield tuple([data_batch, weight_batch])
                else:
                    yield data_batch 
            else:
                if self.use_weights: 
                    yield tuple([x_batch, y_batch, weight_batch])
                else:
                    yield tuple([x_batch, y_batch]) 

    def get_data_for_eval(self):
        X = {}
        Y = {}
        #we don't want to assume that the data is presented in a random order.
        #so we select random batches of the data for evaluation
        num_batches=self.num_to_load_for_eval/self.batch_size + 1
        for input_mode in self.X: 
            X[input_mode]=np.concatenate([self.X[input_mode][self.permuted_batch_start[i]:self.permuted_batch_end[i]] for i in range(num_batches)],axis=0)
        for output_mode in self.Y:
            Y[output_mode]=np.concatenate([self.X[input_mode][self.permuted_batch_start[i]:self.permuted_batch_end[i]] for i in range(num_batches)],axis=0)
        return util.enum(X=X,Y=Y)

    def get_data(self):
        return util.enum(X=self.X, Y=self.Y)

    def get_samples_per_epoch(self):
        return self.num_items
    
class AbstractAtOnceDataLoader(AbstractDataLoader):

    def get_data(self):
        raise NotImplementedError()


class AtOnceDataLoader_XYDictAPI(BatchDataLoader_XYDictAPI):

    def __init__(self, X_full, Y_full,
                       max_to_load=None,
                       #arguments below are only relevant if
                       #want to use in batches as well
                       num_to_load_for_eval=None,
                       bundle_x_and_y_in_generator=None,
                       batch_size=None, 
                       **kwargs):
        self.max_to_load = max_to_load

        X = {}
        Y = {}
        for input_mode in X_full:
            if (self.max_to_load is None):
                X[input_mode] = np.array(X_full[input_mode])
            else:
                X[input_mode] = np.array(X_full[input_mode][:self.max_to_load])
        for output_mode in Y_full:
            if (self.max_to_load is None):
                Y[output_mode] = np.array(Y_full[output_mode])
            else:
                Y[output_mode] = np.array(Y_full[output_mode]
                                                [:self.max_to_load])
        super(AtOnceDataLoader_XYDictAPI, self).__init__(
            X=X, Y=Y,
            weight={}, num_to_load_for_eval=num_to_load_for_eval,
            bundle_x_and_y_in_generator=bundle_x_and_y_in_generator,
            batch_size=batch_size,
            **kwargs)

        self.X = X
        self.Y = Y

    def get_jsonable_object(self):
        the_dict = super(MultimodalAtOnceDataLoader, self)\
                   .get_jsonable_object()
        the_dict['max_to_load'] = self.max_to_load
        return the_dict

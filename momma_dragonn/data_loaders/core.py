from collections import OrderedDict
import numpy as np
from avutils import util
from thread_safety import *
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

    def __init__(self, X, Y, weight, bundle_x_and_y_in_generator,use_weights=True,generator_type='standard',
                       num_to_load_for_eval =None,percent_to_load_for_eval=100, **kwargs):
        super(BatchDataLoader_XYDictAPI, self).__init__(**kwargs)
        self.X = X
        self.Y = Y
        self.weight = weight
        self.use_weights=use_weights
        self.generator_type=generator_type
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
        #handle edge case for very few items:
        if start_index >=self.num_items:
            start_index=0 
        batch_number=-1
        ordered_batches=OrderedDict()
        while start_index < self.num_items:
            batch_number+=1
            end_index=start_index+self.batch_size
            if end_index >=self.num_items:
                end_index=self.num_items
                start_index=self.num_items - self.batch_size
            ordered_batches[batch_number]=[start_index,end_index]
            start_index=start_index+self.batch_size
        #permute the batch indices
        #print("finished permutation") 
        permuted_indices=np.random.permutation(batch_number+1)
        #return start & end indices for the permuted batches
        permuted_batch_start=[]
        permuted_batch_end=[] 
        for i in permuted_indices:
            permuted_batch_start.append(ordered_batches[i][0])
            permuted_batch_end.append(ordered_batches[i][1])
        return permuted_batch_start, permuted_batch_end


    @threadsafe_generator
    def balanced_generator(self,predict_mode=False):
        raise("Balanced generator not implemented!!")
    
    
    #generate at least one item in a batch for each class
    #only relevant for classification tasks (not regression)
    @threadsafe_generator
    def min1_per_class_batch_generator(self,predict_mode=False):
        #generate an inventory of positive & negative values for each task
        positives={}
        negatives={}
        numcolumns={}
        for output_mode in self.output_modes:
            positives[output_mode]=np.argwhere(np.asarray(self.Y[output_mode])==1)
            negatives[output_mode]=np.argwhere(np.asarray(self.Y[output_mode])==0)
            numcolumns[output_mode]=np.shape(self.Y[output_mode])[1]
        
        while True:
            x_batch={}
            y_batch={}
            weight_batch={}
            indices=set()
            for output_mode in self.output_modes:
                numcolumns_cur_output=numcolumns[output_mode]
                for c in range(numcolumns_cur_output):
                    pos_choices=np.argwhere(positives[output_mode][:,1]==c)[0]
                    neg_choices=np.argwhere(negatives[output_mode][:,1]==c)[0]
                    #get one positive for each output-column
                    try:
                        indices.add(positives[output_mode][np.random.choice(pos_choices,1)[0]][0])
                    except:
                        pdb.set_trace() 
                    #get one negative for each output-column
                    indices.add(negatives[output_mode][np.random.choice(neg_choices,1)[0]][0])
            
            num_remaining=self.batch_size - len(indices)
            try:
                start_index=np.random.randint(0,self.num_items - self.batch_size)
            except:
                pdb.set_trace() 
            end_index=start_index+num_remaining
            indices=list(indices)
            indices.sort()
            for input_mode in self.input_modes:
                x_batch[input_mode]=np.concatenate((np.asarray(self.X[input_mode][start_index:end_index]),
                                                    np.asarray(self.X[input_mode][indices])),axis=0)
            for output_mode in self.output_modes:
                y_batch[output_mode]=np.concatenate((np.asarray(self.Y[output_mode][start_index:end_index]),
                                                     np.asarray(self.Y[output_mode][indices])),axis=0)

            if predict_mode==True:
                yield tuple([x_batch,y_batch])
            else:
                if self.use_weights:
                    weight_batch[output_mode]=np.squeeze(np.concatenate((np.asarray(self.Y[output_mode][start_index:end_index]),
                                                              np.asarray(self.Y[output_mode][indices])),axis=0))
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

    
    #this is for the training step, we cycle through all examples
    @threadsafe_generator
    def standard_generator(self,predict_mode=False):   
        self.num_generated=0
        self.permutation_index=0 
        while True:            
            if (self.num_generated>=self.num_items) or (self.permutation_index >= len(self.permuted_batch_start)):
                self.permuted_batch_start,self.permuted_batch_end=self.get_batch_permutation_order()
                self.permutation_index=0
            x_batch = {}
            y_batch = {}
            weight_batch = {}

            #make sure we don't generated more values than samples_per_epoch (aka self.num_items)
            start_index=self.permuted_batch_start[self.permutation_index]
            end_index = self.permuted_batch_end[self.permutation_index]
            for input_mode in self.input_modes:
                x_batch[input_mode] = np.asarray(self.X[input_mode][start_index:end_index])
            for output_mode in self.output_modes:
                y_batch[output_mode] = np.asarray(self.Y[output_mode][start_index:end_index])
            if self.use_weights:
                for output_mode in self.weight:
                    weight_batch[output_mode] = np.squeeze(np.asarray(self.weight[output_mode][start_index:end_index]))

            self.permutation_index+=1
            self.num_generated+=(end_index-start_index)
            if predict_mode==True:
                #the generator will be used to generate values at test time of the model.
                yield tuple([x_batch,y_batch])
            
            elif (self.bundle_x_and_y_in_generator):
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

    @threadsafe_generator
    def get_batch_generator(self,predict_mode=False):
        if self.generator_type=="standard":
            return self.standard_generator(predict_mode)
        elif self.generator_type=="min1_per_class":
            return self.min1_per_class_batch_generator(predict_mode)
        elif self.generator_type=="balanced":
            return self.balanced_generator(predict_mode)
        else:
            raise Exception("invalid type of generator specified!!")

    @threadsafe_generator
    def get_data_for_eval(self):
        self.batches_returned_for_evaluation=0
        self.num_generated=0 
        while True:
            if self.num_generated >= self.num_items:
                yield tuple([{},{}])
            x_batch = {}
            y_batch = {}
            start_index=self.batch_size*self.batches_returned_for_evaluation
            end_index=min([self.num_items,start_index+self.batch_size]) 
            for input_mode in self.input_modes:
                x_batch[input_mode] = np.asarray(self.X[input_mode][start_index:end_index])
            for output_mode in self.output_modes:
                y_batch[output_mode]=np.asarray(self.Y[output_mode][start_index:end_index])
            self.batches_returned_for_evaluation+=1
            self.num_generated+=(end_index-start_index)
            yield tuple([x_batch,y_batch])
        
    def get_data(self):
        return util.enum(X=self.X, Y=self.Y)

    def get_samples_per_epoch(self):
        #This needs to be a multiple of the batch size to avoid warnings about dimensions not matching.
        if self.num_items % self.batch_size !=0:
            return (self.num_items/self.batch_size+1)*self.batch_size
        else:
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

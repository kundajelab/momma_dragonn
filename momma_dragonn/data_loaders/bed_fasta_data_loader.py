from __future__ import print_function
from .core import AbstractBatchDataLoader
import numpy as np
from avutils import util
import pysam
import pandas as pd


class FastaBatchDataLoader(AbstractBatchDataLoader):

    def __init__(self, batch_size, bed_source,
                       rc_augment,
                       ref_fasta,
                       num_to_load_for_eval):
        super(BatchDataLoader, self).__init__(batch_size=batch_size)
        self.bed_source = bed_source
        self.rc_augment = rc_augment
        self.ref_fasta = ref_fasta
        self.num_to_load_for_eval = num_to_load_for_eval
        self.to_load_for_eval_x = []
        self.to_load_for_eval_y = []

    def get_jsonable_object(self):
        the_dict = super(BatchDataLoader, self).get_jsonable_object()
        the_dict['bed_sourc'] = self.bed_source
        the_dict['rc_augment'] = self.rc_augment
        the_dict['ref_fasta'] = self.ref_fasta
        return the_dict

    def get_batch_generator(self):
        #produces the generator for batches 
        #open the reference file 
        ref=pysam.FastaFile(args.ref_fasta) 
        data=pd.read_csv(self.bed_source,
                         header=0,sep='\t',
                         index_col=[0,1,2]) 
        #load the train data as a pandas dataframe, skip the header 
        ltrdict = {
            'a':[1,0,0,0],
            'c':[0,1,0,0],
            'g':[0,0,1,0],
            't':[0,0,0,1],
            'n':[0,0,0,0],
            'A':[1,0,0,0],
            'C':[0,1,0,0],
            'G':[0,0,1,0],
            'T':[0,0,0,1],
            'N':[0,0,0,0]}

        #iterate through batches and one-hot-encode on the fly
        start_index=0 
        last_start_idx=data.shape[0]-(self.batch_size-1)

        while True: 
            if (start_idx > (data.shape[0]-(self.batch_size))): 
                start_index=0 #reset the counter to avoid going over
            end_index=start_index+int(batch_size)
            #get seq positions
            bed_entries=[(data.index[i]) for i
                         in range(start_index,end_index)]
            #get sequences
            seqs=[ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
            if self.rc_augment: 
            seqs=np.array([[ltrdict[x] for x in seq] for seq in seqs])
            if (self.rc_augment):
                seqs = np.concatenate(seqs, seqs[:,::-1,::-1])
                                                                                    
            y_batch=np.asarray(data[start_index:end_index])                         
            if args.revcomp==True:                                                  
                y_batch=np.concatenate((y_batch,y_batch),axis=0)                    
            start_index=end_index                                                   
            assert x_batch.ndim == 3 and y_batch.ndim > 1

            self.to_load_for_eval_x.extend(x_batch)
            self.to_load_for_eval_y.extend(y_batch)
            if (len(self.to_load_for_eval_x) > self.num_to_load_for_eval):
                self.to_load_for_eval_x =\
                    self.to_load_for_eval_x[-self.num_to_load_for_eval:]
                self.to_load_for_eval_y =\
                    self.to_load_for_eval_y[-self.num_to_load_for_eval:]
            
            yield tuple([x_batch,y_batch])     

    def get_data_for_eval(self):
        return util.enum(X=np.array(self.to_load_for_eval_x),
                         Y=np.array(self.to_load_for_eval_y))


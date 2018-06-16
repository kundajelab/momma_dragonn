from __future__ import print_function
from .core import AbstractBatchDataLoader
import numpy as np
from avutils import util


def get_fasta_batch_generator(
    batch_size, ref_fasta, bed_source,
    read_in_order,
    rc_augment, loop_infinitely): 

    import pysam
    import pandas as pd

    #produces the generator for batches 
    #open the reference file 
    ref=pysam.FastaFile(ref_fasta) 
    if (read_in_order):
        data = []
        for a_row in open(bed_source):
            a_row = a_row.rstrip().split("\t")
            data.append(((a_row[0], int(a_row[1]), int(a_row[2])),
                         [int(x) for x in a_row[3:]])) 
    else:
        #load the train data as a pandas dataframe
        data=pd.read_csv(bed_source,
                         header=0,sep='\t',
                         index_col=[0,1,2]) 

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
    if (read_in_order):
        data_length = len(data)
    else:
        data_length = data.shape[0]
    last_start_idx=data_length-(batch_size-1)

    while True: 
        if (start_index > (data_length-(batch_size))): 
            if (loop_infinitely):
                start_index=0 #reset the counter to avoid going over
            else:
                raise StopIteration()
        end_index=start_index+int(batch_size)
        #get seq positions
        if (read_in_order):
            bed_entries = [x[0] for x in data[start_index:end_index]]
        else:
            bed_entries=[(data.index[i]) for i in range(start_index,end_index)]

        #get sequences
        seqs=[ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        seqs=np.array([[ltrdict[x] for x in seq] for seq in seqs])
        if rc_augment:
            seqs = np.concatenate([seqs, seqs[:,::-1,::-1]], axis=0)

        x_batch = seqs
        if (read_in_order):
            y_batch = np.asarray([x[1] for x in data[start_index:end_index]])
        else:
            y_batch=np.asarray(data[start_index:end_index])                         
        if rc_augment: 
            y_batch=np.concatenate((y_batch,y_batch),axis=0) 
        start_index=end_index 
        assert x_batch.ndim == 3 and y_batch.ndim > 1

        yield tuple([x_batch,y_batch,bed_entries]) 


class FastaBatchDataLoader(AbstractBatchDataLoader):

    def __init__(self, batch_size, bed_source,
                       rc_augment,
                       ref_fasta,
                       num_to_load_for_eval,
                       read_in_order,
                       wrap_in_keys=None):
        super(FastaBatchDataLoader, self).__init__(batch_size=batch_size)
        self.bed_source = bed_source
        self.rc_augment = rc_augment
        self.ref_fasta = ref_fasta
        self.num_to_load_for_eval = num_to_load_for_eval
        self.to_load_for_eval_x = []
        self.to_load_for_eval_y = []
        self.read_in_order = read_in_order
        self.wrap_in_keys = wrap_in_keys

    def get_jsonable_object(self):
        the_dict = super(FastaBatchDataLoader, self).get_jsonable_object()
        the_dict['bed_source'] = self.bed_source
        the_dict['rc_augment'] = self.rc_augment
        the_dict['ref_fasta'] = self.ref_fasta
        return the_dict

    def get_batch_generator(self):

        fasta_batch_generator = get_fasta_batch_generator(
            batch_size=self.batch_size, ref_fasta=self.ref_fasta,
            bed_source=self.bed_source, rc_augment=self.rc_augment,
            loop_infinitely=True, read_in_order=self.read_in_order)

        while True:
            x_batch, y_batch, bed_entries = fasta_batch_generator.next()
            self.to_load_for_eval_x.extend(x_batch)
            self.to_load_for_eval_y.extend(y_batch)
            if (len(self.to_load_for_eval_x) > self.num_to_load_for_eval):
                self.to_load_for_eval_x =\
                    self.to_load_for_eval_x[-self.num_to_load_for_eval:]
                self.to_load_for_eval_y =\
                    self.to_load_for_eval_y[-self.num_to_load_for_eval:]

            if (self.wrap_in_keys is not None):
                yield ({self.wrap_in_keys[0]: x_batch},
                       {self.wrap_in_keys[1]: y_batch})
            else:
                yield (x_batch, y_batch)        

    def get_data_for_eval(self):
        if (self.wrap_in_keys is not None):
            return util.enum(
                X={self.wrap_in_keys[0]: np.array(self.to_load_for_eval_x)},
                Y={self.wrap_in_keys[1]: np.array(self.to_load_for_eval_y)})
        else:
            return util.enum(X=np.array(self.to_load_for_eval_x),
                             Y=np.array(self.to_load_for_eval_y))

    def get_data(self):
        fasta_batch_generator = get_fasta_batch_generator(
            batch_size=self.batch_size, ref_fasta=self.ref_fasta,
            bed_source=self.bed_source, rc_augment=self.rc_augment,
            loop_infinitely=False, read_in_order=self.read_in_order)
        X = []
        Y = []
        for (x_batch, y_batch, bed_entries) in fasta_batch_generator:
            X.extend(x_batch)
            Y.extend(y_batch)
        if (self.wrap_in_keys is not None):
            return util.enum(X={self.wrap_in_keys[0]: np.array(X)},
                             Y={self.wrap_in_keys[1]: np.array(Y)})
        else:
            return util.enum(X=np.array(X), Y=np.array(Y))



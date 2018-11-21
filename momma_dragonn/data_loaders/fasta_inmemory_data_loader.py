from __future__ import division, absolute_import, print_function
from .core import AbstractBatchDataLoader
import numpy as np
from avutils import util
from avutils import file_processing as fp
from collections import namedtuple


one_hot_encode = {
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

class AbstractSeqOnlyDataLoader(AbstractBatchDataLoader):


    def __init__(self, batch_size,
                       rc_augment,
                       num_to_load_for_eval,
                       wrap_in_keys=None):
        super(AbstractSeqOnlyDataLoader, self).__init__(batch_size=batch_size)
        self.rc_augment = rc_augment
        self.num_to_load_for_eval = num_to_load_for_eval
        self.to_load_for_eval_x = []
        self.to_load_for_eval_y = []
        self.wrap_in_keys = wrap_in_keys

    def get_jsonable_object(self):
        the_dict = super(AbstractSeqOnlyDataLoader, self).get_jsonable_object()
        the_dict['rc_augment'] = self.rc_augment
        the_dict['num_to_load_for_eval'] = self.num_to_load_for_eval
        the_dict['wrap_in_keys'] = self.wrap_in_keys
        return the_dict

    def get_generator(self, loop_infinitely):
        raise NotImplementedError()

    def get_batch_generator(self):

        fasta_generator = self.get_generator(loop_infinitely=True)

        while True:
            x_batch = []
            y_batch = []
            for i in range(self.batch_size):
                x,y = fasta_generator.next()
                x_batch.append(x)
                y_batch.append(y)
                if (self.rc_augment):
                    x_batch.append(x[::-1,::-1])
                    y_batch.append(y)
            if (len(set([len(x) for x in x_batch])) > 1):
                #weed out examples that are shorter than they should be 
                max_len = max([len(x) for x in x_batch])
                new_x = []
                new_y = []
                for x,y in zip(x_batch, y_batch):
                    if (len(x) == max_len):
                        new_x.append(x)
                        new_y.append(y) 
                    else:
                        print("Skipping line with length",len(x))
                x_batch = new_x
                y_batch = new_y
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
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
        fasta_generator = self.get_generator(loop_infinitely=False)
        X = []
        Y = []
        for (x,y) in fasta_generator:
            X.append(x)
            Y.append(y)
        if (self.wrap_in_keys is not None):
            return util.enum(X={self.wrap_in_keys[0]: np.array(X)},
                             Y={self.wrap_in_keys[1]: np.array(Y)})
        else:
            return util.enum(X=np.array(X), Y=np.array(Y))


def get_fastaseq_generator(file_with_fasta, fasta_col,
                           randomize_after_pass,
                           random_seed, loop_infinitely,
                           label_columns, labels_dtype):
    #read bed_source into memory
    bed_fh = fp.get_file_handle(file_with_fasta)
    data = []
    print("Reading file "+file_with_fasta+" into memory")
    for a_row in bed_fh:
        a_row = a_row.rstrip().split("\t")
        data.append((a_row[fasta_col], [labels_dtype(a_row[x]) for
                                        x in label_columns]))
    print("Finished reading file into memory; got "
          +str(len(data))+"rows")
    random_obj = np.random.RandomState(random_seed)
    if (randomize_after_pass):
        data = shuffle_array(arr=data, random_obj=random_obj)

    idx = 0
    while (idx < len(data)):
        to_yield = np.array([one_hot_encode[x] for x in data[idx]])
        yield to_yield
        idx += 1
        if (idx==len(data)):
            if (loop_infinitely):
                if (randomize_after_pass):
                    data = shuffle_array(arr=data, random_obj=random_obj)
                idx=0
            else:
                raise StopIteration()


class TwoStreamSeqOnly(AbstractSeqOnlyDataLoader):

    def __init__(self, batch_size,
                       positives_fasta_source,
                       negatives_fasta_source,
                       fasta_col,
                       negatives_to_positives_ratio,
                       rc_augment,
                       num_to_load_for_eval,
                       label_columns=[],
                       randomize_after_pass=True,
                       random_seed=1,
                       labels_dtype="int",
                       wrap_in_keys=None):
        super(TwoStreamSeqOnly, self).__init__(
            batch_size=batch_size,
            rc_augment=rc_augment,
            num_to_load_for_eval=num_to_load_for_eval,
            wrap_in_keys=wrap_in_keys)
        self.positives_fasta_source = positives_fasta_source
        self.negatives_fasta_source = negatives_fasta_source
        self.fasta_col = fasta_col
        self.negatives_to_positives_ratio = negatives_to_positives_ratio
        assert isinstance(negatives_to_positives_ratio, int)
        self.randomize_after_pass = randomize_after_pass
        self.random_seed = random_seed
        self.str_labels_dtype = labels_dtype
        self.labels_dtype=eval(labels_dtype)
        self.label_columns = label_columns

    def get_jsonable_object(self):
        the_dict = super(TwoStreamSeqOnly, self).get_jsonable_object()
        the_dict['positives_fasta_source'] = self.positives_fasta_source
        the_dict['negatives_fasta_source'] = self.negatives_fasta_source
        the_dict['fasta_col'] = self.fasta_col
        the_dict['randomize_after_pass'] = self.randomize_after_pass
        the_dict['random_seed'] = self.random_seed
        the_dict['labels_dtype'] = self.str_labels_dtype
        the_dict['label_columns'] = label_columns
        
        return the_dict

    def get_generator(self, loop_infinitely):
        assert loop_infinitely==True,\
            "TwoStream not supported for no infinite looping"
        positives_generator = get_fastaseq_generator(
                    file_with_fasta=self.positives_fasta_source,
                    fasta_col=self.fasta_col,
                    randomize_after_pass=self.randomize_after_pass,
                    random_seed=self.random_seed,
                    loop_infinitely=loop_infinitely,
                    label_columns=self.label_columns,
                    labels_dtype=self.labels_dtype)
        negatives_generator = get_fastaseq_generator(
                    file_with_fasta=self.negatives_fasta_source,
                    fasta_col=self.fasta_col,
                    randomize_after_pass=self.randomize_after_pass,
                    random_seed=self.random_seed,
                    loop_infinitely=loop_infinitely)
        while 1:
            to_yield = (positives_generator.next()
                        if hasattr(positives_generator, 'next')
                        else positives_generator.__next__())
            yield (to_yield if len(to_yield[1]) > 0 else (to_yield, [1.0]))
            for i in range(self.negatives_to_positives_ratio):
                to_yield = (negatives_generator.next()
                            if hasattr(negatives_generator, 'next')
                            else negatives_generator.__next__())
                yield (to_yield if len(to_yield[1]) > 0 else (to_yield, [0.0]))


#randomly shuffles the input array
#mutates arr
def shuffle_array(arr, random_obj):
    for i in range(0,len(arr)-1):
        #randomly select index:
        chosen_index = random_obj.randint(i,len(arr)-1)
        val_at_index = arr[chosen_index]
        arr[chosen_index] = arr[i]
        arr[i] = val_at_index
    return arr

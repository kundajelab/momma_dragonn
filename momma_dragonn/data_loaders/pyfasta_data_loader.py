from __future__ import division, absolute_import, print_function
from .core import AbstractBatchDataLoader
import numpy as np
from avutils import util
from avutils import file_processing as fp
from collections import namedtuple


Interval = namedtuple("Interval", ["chrom", "start", "stop", "labels"])
one_hot_encode = {
        'a':np.array([1,0,0,0]),
        'c':np.array([0,1,0,0]),
        'g':np.array([0,0,1,0]),
        't':np.array([0,0,0,1]),
        'n':np.array([0,0,0,0]),
        'A':np.array([1,0,0,0]),
        'C':np.array([0,1,0,0]),
        'G':np.array([0,0,1,0]),
        'T':np.array([0,0,0,1]),
        'N':np.array([0,0,0,0])}

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
                if (hasattr(fasta_generator, 'next')):
                    x,y,coor = fasta_generator.next()
                else:
                    x,y,coor = fasta_generator.__next__()
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
                        print("Skipping",coor,"with length",len(x))
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
        for (x,y,coor) in fasta_generator:
            X.append(x)
            Y.append(y)
        if (self.wrap_in_keys is not None):
            return util.enum(X={self.wrap_in_keys[0]: np.array(X)},
                             Y={self.wrap_in_keys[1]: np.array(Y)})
        else:
            return util.enum(X=np.array(X), Y=np.array(Y))


def get_pyfasta_generator(bed_source, fasta_data_source,
                          append_chrom_number, labels_dtype,
                          randomize_after_pass,
                          random_seed, loop_infinitely):
    #read bed_source into memory
    bed_fh = fp.get_file_handle(bed_source)
    data = []
    print("Reading bed file "+bed_source+" into memory")
    for a_row in bed_fh:
        a_row = a_row.decode("utf-8").rstrip().split("\t")
        data.append(Interval(
            chrom=a_row[0], start=int(a_row[1]), stop=int(a_row[2]),
            labels=[labels_dtype(x) for x in a_row[3:]]))
    print("Finished reading bed file into memory; got "
          +str(len(data))+"rows")
    random_obj = np.random.RandomState(random_seed)
    if (randomize_after_pass):
        data = shuffle_array(arr=data, random_obj=random_obj)

    #fasta extraction
    import pyfasta
    f = pyfasta.Fasta(fasta_data_source)

    idx = 0
    while (idx < len(data)):
        to_extract = data[idx:idx+1]
        chrom = to_extract[0].chrom
        if (append_chrom_number == True):
            chrom = chrom+" "+chrom[3:]
        to_yield = f[chrom][to_extract[0].start:to_extract[0].stop]
        to_yield = np.array([one_hot_encode[x] for x in to_yield])
        yield (to_yield, to_extract[0].labels,
               (to_extract[0].chrom,
                to_extract[0].start,
                to_extract[0].stop))

        idx += 1
        if (idx==len(data)):
            if (loop_infinitely):
                if (randomize_after_pass):
                    data = shuffle_array(arr=data, random_obj=random_obj)
                idx=0
            else:
                raise StopIteration()


class SingleStreamSeqOnly(AbstractSeqOnlyDataLoader):

    def __init__(self, batch_size,
                       bed_source,
                       fasta_data_source,
                       rc_augment,
                       num_to_load_for_eval,
                       randomize_after_pass=True,
                       random_seed=1,
                       labels_dtype="int",
                       wrap_in_keys=None,
                       append_chrom_number=False):
        super(SingleStreamSeqOnly, self).__init__(
            batch_size=batch_size,
            rc_augment=rc_augment,
            num_to_load_for_eval=num_to_load_for_eval,
            wrap_in_keys=wrap_in_keys)
        self.bed_source = bed_source
        self.fasta_data_source = fasta_data_source
        self.str_labels_dtype = labels_dtype
        self.randomize_after_pass = randomize_after_pass
        self.random_seed = random_seed
        self.labels_dtype=eval(labels_dtype)
        self.append_chrom_number = append_chrom_number

    def get_jsonable_object(self):
        the_dict = super(SingleStreamSeqOnly, self).get_jsonable_object()
        the_dict['bed_source'] = self.bed_source
        the_dict['fasta_data_source'] = self.fasta_data_source
        the_dict['labels_dtype'] = self.str_labels_dtype
        the_dict['randomize_after_pass'] = self.randomize_after_pass
        the_dict['random_seed'] = self.random_seed
        return the_dict

    def get_generator(self, loop_infinitely):
        return get_pyfasta_generator(bed_source=self.bed_source,
                              fasta_data_source=self.fasta_data_source,
                              append_chrom_number=self.append_chrom_number,
                              labels_dtype=self.labels_dtype,
                              randomize_after_pass=self.randomize_after_pass,
                              random_seed=self.random_seed,
                              loop_infinitely=loop_infinitely)


class TwoStreamSeqOnly(AbstractSeqOnlyDataLoader):

    def __init__(self, batch_size,
                       positives_bed_source,
                       negatives_bed_source,
                       negatives_to_positives_ratio,
                       fasta_data_source,
                       rc_augment,
                       num_to_load_for_eval,
                       randomize_after_pass=True,
                       random_seed=1,
                       labels_dtype="int",
                       wrap_in_keys=None,
                       append_chrom_number=False):
        super(TwoStreamSeqOnly, self).__init__(
            batch_size=batch_size,
            rc_augment=rc_augment,
            num_to_load_for_eval=num_to_load_for_eval,
            wrap_in_keys=wrap_in_keys)
        self.positives_bed_source = positives_bed_source
        self.negatives_bed_source = negatives_bed_source
        self.negatives_to_positives_ratio = negatives_to_positives_ratio
        assert isinstance(negatives_to_positives_ratio, int)
        self.fasta_data_source = fasta_data_source
        self.str_labels_dtype = labels_dtype
        self.randomize_after_pass = randomize_after_pass
        self.random_seed = random_seed
        self.labels_dtype=eval(labels_dtype)
        self.append_chrom_number = append_chrom_number

    def get_jsonable_object(self):
        the_dict = super(TwoStreamSeqOnly, self).get_jsonable_object()
        the_dict['positives_bed_source'] = self.positives_bed_source
        the_dict['negatives_bed_source'] = self.negatives_bed_source
        the_dict['fasta_data_source'] = self.fasta_data_source
        the_dict['labels_dtype'] = self.str_labels_dtype
        the_dict['randomize_after_pass'] = self.randomize_after_pass
        the_dict['random_seed'] = self.random_seed
        return the_dict

    def get_generator(self, loop_infinitely):
        assert loop_infinitely==True,\
            "TwoStream not supported for no infinite looping"
        positives_generator = get_pyfasta_generator(
                    bed_source=self.positives_bed_source,
                    fasta_data_source=self.fasta_data_source,
                    append_chrom_number=self.append_chrom_number,
                    labels_dtype=self.labels_dtype,
                    randomize_after_pass=self.randomize_after_pass,
                    random_seed=self.random_seed,
                    loop_infinitely=loop_infinitely)
        negatives_generator = get_pyfasta_generator(
                    bed_source=self.negatives_bed_source,
                    fasta_data_source=self.fasta_data_source,
                    append_chrom_number=self.append_chrom_number,
                    labels_dtype=self.labels_dtype,
                    randomize_after_pass=self.randomize_after_pass,
                    random_seed=self.random_seed,
                    loop_infinitely=loop_infinitely)
        while 1:
            if (hasattr(positives_generator, 'next')):
                yield positives_generator.next() 
            else:
                yield positives_generator.__next__()
            for i in range(self.negatives_to_positives_ratio):
                if (hasattr(negatives_generator, 'next')):
                    yield negatives_generator.next()
                else:
                    yield negatives_generator.__next__()


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

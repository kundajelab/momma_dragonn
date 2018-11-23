from __future__ import division, absolute_import, print_function
from .core import AbstractBatchDataLoader
import numpy as np
from avutils import util
from avutils import file_processing as fp
from collections import namedtuple


Interval = namedtuple("Interval", ["chrom", "start", "stop", "labels"])
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
rc_trans = {'a':'t', 'c':'g', 'g':'c', 't':'a', 'n':'n',
            'A':'T', 'C':'G', 'G':'C', 'T':'A', 'N':'N'}

class AbstractSeqOnlyDataLoader(AbstractBatchDataLoader):


    def __init__(self, batch_size,
                       rc_augment,
                       num_to_load_for_eval,
                       wrap_in_keys=None):
        super(AbstractSeqOnlyDataLoader, self).__init__(batch_size=batch_size)
        self.rc_augment = rc_augment
        self.num_to_load_for_eval = num_to_load_for_eval
        self.to_load_for_eval_coors = []
        self.to_load_for_eval_fastastrs = []
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
            coor_batch = []
            x_batch = []
            y_batch = []
            fastastr_batch = []
            for i in range(self.batch_size):
                if (hasattr(fasta_generator, 'next')):
                    x,y,coor,fastastr = fasta_generator.next()
                else:
                    x,y,coor,fastastr = fasta_generator.__next__()
                fastastr_batch.append(fastastr)
                coor_batch.append(coor)
                x_batch.append(x)
                y_batch.append(y)
                if (self.rc_augment):
                    fastastr_batch.append("".join([rc_trans[l]
                                                   for l in fastastr]))
                    x_batch.append(x[::-1,::-1])
                    y_batch.append(y)
                    coor_batch.append(coor)
            if (len(set([len(x) for x in x_batch])) > 1):
                #weed out examples that are shorter than they should be 
                max_len = max([len(x) for x in x_batch])
                new_fastastr = []
                new_coor = []
                new_x = []
                new_y = []
                for fastastr,coor,x,y in zip(fastastr_batch, coor_batch,
                                             x_batch, y_batch):
                    if (len(x) == max_len):
                        new_fastastr.append(fastastr)
                        new_coor.append(coor)
                        new_x.append(x)
                        new_y.append(y) 
                    else:
                        print("Skipping",coor,"with length",len(x))
                fastastr_batch = new_fastastr
                coor_batch = new_coor
                x_batch = new_x
                y_batch = new_y
            assert len(x_batch)==len(y_batch)
            self.to_load_for_eval_fastastrs.extend(fastastr_batch)
            self.to_load_for_eval_coors.extend(coor_batch)
            self.to_load_for_eval_x.extend(x_batch)
            self.to_load_for_eval_y.extend(y_batch)
            if (len(self.to_load_for_eval_x) > self.num_to_load_for_eval):
                self.to_load_for_eval_fastastrs = self.to_load_for_eval_fastastrs[-self.num_to_load_for_eval:]
                self.to_load_for_eval_coors = self.to_load_for_eval_coors[-self.num_to_load_for_eval:]
                self.to_load_for_eval_x =\
                    self.to_load_for_eval_x[-self.num_to_load_for_eval:]
                self.to_load_for_eval_y =\
                    self.to_load_for_eval_y[-self.num_to_load_for_eval:]
                assert np.max(np.abs(
                              np.array(self.to_load_for_eval_x[-self.num_to_load_for_eval:])-
                              np.array(self.to_load_for_eval_x[len(self.to_load_for_eval_x)-self.num_to_load_for_eval:len(self.to_load_for_eval_x)])))==0
                assert np.max(np.abs(
                              np.array(self.to_load_for_eval_y[-self.num_to_load_for_eval:])-
                              np.array(self.to_load_for_eval_y[len(self.to_load_for_eval_x)-self.num_to_load_for_eval:len(self.to_load_for_eval_x)])))==0
            assert len(self.to_load_for_eval_x)==len(self.to_load_for_eval_y)

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
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
            #Studying weird side-effect...
            #print("x",len(self.to_load_for_eval_x),
            #      "y",len(self.to_load_for_eval_y))
            #print("x",len(self.to_load_for_eval_x),
            #      "y",len(self.to_load_for_eval_y))
            #print("Call np.array on y")
            #np.array(self.to_load_for_eval_y)
            #print("x",len(self.to_load_for_eval_x),
            #      "y",len(self.to_load_for_eval_y))
            #print("x",len(self.to_load_for_eval_x),
            #      "y",len(self.to_load_for_eval_y))
            #print("Call np.array on y")
            #np.array(self.to_load_for_eval_y)
            #print("x",len(self.to_load_for_eval_x),
            #      "y",len(self.to_load_for_eval_y))
            #print("x",len(self.to_load_for_eval_x),
            #      "y",len(self.to_load_for_eval_y))
            #print("Call np.array on x")
            #np.array(self.to_load_for_eval_x)
            #print("x",len(self.to_load_for_eval_x),
            #      "y",len(self.to_load_for_eval_y))
            #print("x",len(self.to_load_for_eval_x),
            #      "y",len(self.to_load_for_eval_y))
            #print("Call np.array on x")
            #np.array(self.to_load_for_eval_x)
            #print("x",len(self.to_load_for_eval_x),
            #      "y",len(self.to_load_for_eval_y))
            #print("x",len(self.to_load_for_eval_x),
            #      "y",len(self.to_load_for_eval_y))

            assert len(np.array(self.to_load_for_eval_x))==len(np.array(self.to_load_for_eval_y))
            return util.enum(X=np.array(self.to_load_for_eval_x),
                             Y=np.array(self.to_load_for_eval_y),
                             coors=self.to_load_for_eval_coors,
                             fastastr=self.to_load_for_eval_fastastrs)

    def get_data(self):
        fasta_generator = self.get_generator(loop_infinitely=False)
        X = []
        Y = []
        for (x,y,coor,fastastr) in fasta_generator:
            X.append(x)
            Y.append(y)
        if (self.wrap_in_keys is not None):
            return util.enum(X={self.wrap_in_keys[0]: np.array(X)},
                             Y={self.wrap_in_keys[1]: np.array(Y)})
        else:
            return util.enum(X=np.array(X), Y=np.array(Y))


def get_stratified_shuffle(stratifications, random_obj):
    stratifications = [
        shuffle_array(arr=this_split,
                      random_obj=random_obj)
        for this_split in stratifications] 
    shuffled_data = []
    idx_within_split = 0
    while len(shuffled_data) < sum([len(x) for x in stratifications]):
        for split_idx in range(len(stratifications)):
            if len(stratifications[split_idx]) > idx_within_split:
                shuffled_data.append(
                    stratifications[split_idx][idx_within_split]) 
        idx_within_split += 1
    return shuffled_data


def get_pyfasta_generator(bed_source, fasta_data_source,
                          append_chrom_number, labels_dtype,
                          randomize_after_pass,
                          stratification_settings,
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

    if (stratification_settings is not None):
        stratification_type = stratification_settings["type"] 
        stratification_column = stratification_settings["column"]
        num_splits = stratification_settings['num_splits']
        bin_sizes = int(np.ceil(len(data)/num_splits))
        if (stratification_type=="continuous"):
            sorted_data = sorted(
                data, key=lambda x: x.labels[stratification_column]) 
            stratifications = [
                sorted_data[i*bin_sizes:
                            min(len(data), (i+1)*bin_sizes)]
                for i in range(num_splits)
            ] 
        else:
            raise RuntimeError(
                "Unrecognized stratification type",
                stratification_type)

    if (randomize_after_pass):
        if (stratification_settings is not None):
            data = get_stratified_shuffle(
                    stratifications=stratifications,
                    random_obj=random_obj)
        else:
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
        to_yield_str = f[chrom][to_extract[0].start:to_extract[0].stop]
        to_yield = np.array([one_hot_encode[x] for x in to_yield_str])
        yield (to_yield, to_extract[0].labels,
               (to_extract[0].chrom,
                to_extract[0].start,
                to_extract[0].stop),
               to_yield_str)

        idx += 1
        if (idx==len(data)):
            if (loop_infinitely):
                if (randomize_after_pass):
                    if (stratification_settings is not None):
                        data = get_stratified_shuffle(
                            stratifications=stratifications,
                            random_obj=random_obj)
                    else:
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
                       stratification_settings=None,
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
        self.stratification_settings = stratification_settings
        self.random_seed = random_seed
        self.labels_dtype=eval(labels_dtype)
        self.append_chrom_number = append_chrom_number

    def get_jsonable_object(self):
        the_dict = super(SingleStreamSeqOnly, self).get_jsonable_object()
        the_dict['bed_source'] = self.bed_source
        the_dict['fasta_data_source'] = self.fasta_data_source
        the_dict['labels_dtype'] = self.str_labels_dtype
        the_dict['randomize_after_pass'] = self.randomize_after_pass
        the_dict['stratification_settings'] = self.stratification_settings
        the_dict['random_seed'] = self.random_seed
        return the_dict

    def get_generator(self, loop_infinitely):
        return get_pyfasta_generator(
                    bed_source=self.bed_source,
                    fasta_data_source=self.fasta_data_source,
                    append_chrom_number=self.append_chrom_number,
                    labels_dtype=self.labels_dtype,
                    randomize_after_pass=self.randomize_after_pass,
                    stratification_settings=self.stratification_settings,
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
                       stratification_settings=None,
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
        self.stratification_settings = stratification_settings
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
        the_dict['stratification_settings'] = self.stratification_settings
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
                    stratification_settings=self.stratification_settings,
                    random_seed=self.random_seed,
                    loop_infinitely=loop_infinitely)
        negatives_generator = get_pyfasta_generator(
                    bed_source=self.negatives_bed_source,
                    fasta_data_source=self.fasta_data_source,
                    append_chrom_number=self.append_chrom_number,
                    labels_dtype=self.labels_dtype,
                    randomize_after_pass=self.randomize_after_pass,
                    stratification_settings=self.stratification_settings,
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

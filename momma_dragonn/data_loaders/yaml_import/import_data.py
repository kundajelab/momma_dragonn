import sys
import os
from collections import namedtuple, OrderedDict, defaultdict
import itertools
import numpy as np
import avutils.file_processing as fp
import avutils.util as av_util
from avutils.dynamic_enum import Key, Keys
import h5py


ContentType=namedtuple('ContentType',['name','casting_function'])
ContentTypes=av_util.enum(integer=ContentType("int",int),
                       floating=ContentType("float",float),
                       string=ContentType("str",str))

ContentTypesLookup = dict((x.name,x.casting_function)
                          for x in ContentTypes.vals)

#TODO: implement weights
RootKeys=Keys(Key("features"), Key("labels"), Key("splits"), Key("weights"))

###
#Features Keys
###
FeaturesFormat=av_util.enum(rows_and_columns='rows_and_columns'
                         , fasta='fasta')
DefaultModeNames = av_util.enum(labels="default_output_mode_name",
                             features="default_input_mode_name")
FeaturesKeys = Keys(Key("features_format")
                    , Key("opts")
                    , Key("input_mode_name",
                    default=DefaultModeNames.features))
FeatureSetYamlKeys_RowsAndCols = Keys(
    Key("file_names")
    ,Key("content_type",default=ContentTypes.floating.name)
    ,Key("content_start_index",default=1)
    ,Key("subset_of_columns_to_use_options",default=None)
    ,Key("progress_update",default=None));
SubsetOfColumnsToUseOptionsYamlKeys = Keys(
                                   Key("subset_of_columns_to_use_mode"),
                                   Key("file_with_column_names", default=None),
                                   Key("N", default=None))
###
#options for FeaturesFormat
###
FeaturesFormat = av_util.enum(rowsAndColumns='rowsAndColumns'
                         , fasta='fasta')
#For files that have the format produced by getfasta bedtools;
#>key \n [fasta sequence] \n ...
FeaturesFormatOptions_Fasta = Keys(Key("file_names"),
                                   Key("progress_update", default=None))


###
#Label keys
###
LabelsKeys = Keys(Key("file_name"),
                  Key("content_type", default=ContentTypes.integer.name),
                  Key("file_with_subset_of_label_names", default=None),
                  Key("output_mode_name", default=DefaultModeNames.labels),
                  Key("progress_update", default=None),
		  Key("key_columns", default=[0]))

###
#Weight Keys
###
WeightsKeys=Keys(Key("weights"), Key("output_mode_name",
                                     default=DefaultModeNames.labels)) 

###
#Split keys
###
SplitOptsKeys = Keys(Key("title_present",default=False),
                     Key("col",default=0))
SplitKeys = Keys(Key("split_name_to_split_files"),
                 Key("opts",
                     default=SplitOptsKeys.fill_in_defaults_for_keys({})))


class AbstractDataForSplitCompiler(object):

    def __init__(self, split_name, ids_in_split):
        self.split_name = split_name
        self.ids_in_split = ids_in_split
        self.output_mode_to_label_names = OrderedDict()

    def add_features(self, the_id, input_mode, features):
        raise NotImplementedError() 

    def set_label_names(self, output_mode, label_names):
        self.output_mode_to_label_names[output_mode] = label_names

    def add_labels(self, the_id, output_mode, labels):
        raise NotImplementedError()

    def add_weights(self, the_id, sample_weights, output_mode): 
        raise NotImplementedError()


class InMemoryDataForSplitCompiler(AbstractDataForSplitCompiler):

    def __init__(self, **kwargs):
        super(InMemoryDataForSplitCompiler, self).__init__(**kwargs)
        self.input_mode_to_id_to_features =\
            av_util.DefaultOrderedDictWrapper(factory={})
        self.output_mode_to_id_to_features =\
            av_util.DefaultOrderedDictWrapper(factory={})
        self.output_mode_to_id_to_weights =\
            av_util.DefaultOrderedDictWrapper(factory={})

    def add_features(self, the_id, features, input_mode):
        self.input_mode_to_id_to_features[input_mode][the_id] = features

    def add_labels(self, the_id, labels, output_mode):
        self.output_mode_to_id_to_labels[output_mode][the_id] = labels

    def add_weights(self, the_id, weights, output_mode):
        self.output_mode_to_id_to_weights[output_mode][the_id] = weights

    def finalize(self):
        for attr in ['X','Y','weights']:
            setattr(self, attr, DefaultOrderedDictWrapper(factory=[]))

        for the_id in self.ids_in_split:
            if (self.check_in_everything(the_id)):
                for mode_to_array, mode_to_id_to_val in [
                    (self.X, self.input_mode_to_id_to_features),
                    (self.Y, self.output_mode_name_to_id_to_labels),
                    (self.weights, self.output_mode_to_id_to_weights)]:
                    self.append_to_array_using_id(the_id, mode_to_array,
                                                  mode_to_id_to_val)

        for attr in ['X','Y','weights']:
            #set the attributes to the underlying ordered_dict of
            #DefaultOrderedDictWrapper
            setattr(self, attr, getattr(self, attr).ordered_dict)

    @staticmethod
    def append_to_array_using_id(the_id, mode_to_array, mode_to_id_to_val):
        #conveniently enough, this loop doesn't throw an error
        #if mode_to_id_to_val is empty (as could happen if the
        #weights are not specified)
        for mode in mode_to_id_to_val:
            mode_to_array[mode].append(mode_to_id_to_val[mode][the_id]) 
 
    def check_in_everything(self, the_id):
        for parent_dict, parent_dict_name in (
            (self.input_mode_to_id_to_features, 'features'),
            (self.output_mode_to_id_to_features, 'outputs'),
            (self.output_mode_to_id_to_weights, 'weights')):
            #conveniently enough, this loop doesn't throw an error
            #if parent_dict is completely empty (as may happen if
            #weights are not specified)
            for mode in parent_dict:
                if the_id not in parent_dict[mode]:
                    print("Warning:",the_id,"absent from",
                          parent_dict_name,"mode",mode) 
                    return False
        return True


class Hdf5ForSplitCompiler(AbstractDataForSplitCompiler):

    def __init__(self, output_dir=".", **kwargs):
        super(Hdf5ForSplitCompiler, self).__init__(**kwargs)
        #create the file to store the features, labels and weights
        self.f = h5py.File(output_dir+"/"+self.split_name+"_data.hdf5","w") 
        self.num_ids = len(self.ids_in_split)
        self.X = self.f.create_group("X")
        self.Y = self.f.create_group("Y")
        self.weights = self.f.create_group("weights")

        self.features_input_mode_to_count = defaultdict(lambda: 0) 
        self.labels_output_mode_to_count = defaultdict(lambda: 0) 
        self.weights_output_mode_to_count = defaultdict(lambda: 0) 

        self.features_input_mode_to_buffer = {}
        self.labels_output_mode_to_buffer = {}
        self.weights_output_mode_to_buffer = {}

    def add_features(self, the_id, features, input_mode):
        self.extend_dataset(
            mode=input_mode,
            the_id=the_id,
            vals=features,
            mode_to_count=self.features_input_mode_to_count,
            mode_to_buffer=self.features_input_mode_to_buffer,
            hdf5grp=self.X,
            shape=[x for x in features.shape])

    def add_labels(self, the_id, labels, output_mode):
        self.extend_dataset(
            mode=output_mode,
            the_id=the_id,
            vals=labels,
            mode_to_count=self.labels_output_mode_to_count,
            mode_to_buffer=self.labels_output_mode_to_buffer,
            hdf5grp=self.Y,
            shape=[len(labels)])

    def add_weights(self, the_id, weights, output_mode):
        self.extend_dataset(
            mode=output_mode,
            the_id=the_id,
            vals=weights,
            mode_to_count=self.weights_output_mode_to_count,
            mode_to_buffer=self.weights_output_mode_to_buffer,
            hdf5grp=self.weights,
            shape=[])

    def extend_dataset(self, mode, the_id, vals,
                       mode_to_count, mode_to_buffer,
                       hdf5grp, shape):
        idx = mode_to_count[mode]
        assert self.ids_in_split[idx] == the_id,\
               ("On index "+str(idx)+"; id meant to be "
                +the_id+" but got "+self.ids_in_split[idx])
        if (idx == 0):
            dset = hdf5grp.create_dataset(mode, [self.num_ids]+shape,
                                          compression="gzip") 
            mode_to_buffer[mode] = fp.Hdf5BufferedDatasetWriter(dset)
        mode_to_buffer[mode].write(vals)
        mode_to_count[mode] += 1

    def finalize(self):
        for mode_to_buffer in [
            self.features_input_mode_to_buffer,
            self.labels_output_mode_to_buffer,
            self.weights_output_mode_to_buffer]:
            for mode in mode_to_buffer:
                mode_to_buffer[mode].flush() 

    def close(self):
        self.f.close()


def process_series_of_yamls(series_of_yamls, split_compiler_factory):
    #if some of the things in seriesOfYamls are lists, chain together
    series_of_yamls = list(itertools.chain(
                        *[x if isinstance(x,list) else [x]
                        for x in series_of_yamls]))
    combined_yaml = get_combined_yaml_from_series_of_yamls(series_of_yamls)
    return process_combined_yamls(combined_yaml, split_compiler_factory)


def get_combined_yaml_from_series_of_yamls(series_of_yamls):
    #if there are, for instance, multiple feature yamls and/or
    #multiple label yamls, will condense them.
    combined_yaml = OrderedDict([('features',[]),
                                 ('labels',[]),
                                 ('weights',[])])
    #the contents of the yaml file must be a dictionaries where
    #the keys at the root are one of RootKeys.keys, which are
    #at the time of writing: 'features', 'labels' or 'splits' 
    for yaml_object in series_of_yamls:
        RootKeys.check_for_unsupported_keys(yaml_object)
        for key in [RootKeys.keys.features,
                    RootKeys.keys.labels,
                    RootKeys.keys.weights]:
            if (key in yaml_object):
                combined_yaml[key].append(yaml_object[key])
        for key in [RootKeys.keys.splits]:
            if key in yaml_object:
                if key not in combined_yaml:
                    combined_yaml[key] = yaml_object[key]
                else:
                    raise RuntimeError("Two specifications given for "
                                       +str(key))
    return combined_yaml


def process_combined_yamls(combined_yamls, split_compiler_factory):
    #get the splits info 
    id_to_split_names, split_to_ids = get_id_to_split_names(
                                          combined_yamls[RootKeys.keys.splits])

    #initialize the data for split compiler objects using
    #the split information
    split_name_to_compiler = OrderedDict()
    for split_name in split_to_ids:
        split_name_to_compiler[split_name] =\
            split_compiler_factory(split_name=split_name,
                                    ids_in_split=split_to_ids[split_name]) 

    #define the action that will get applied to new labels
    def labels_action(output_mode, the_id, labels): 
        split_names = id_to_split_names[the_id]
        for split_name in split_names:
            split_name_to_compiler[split_name].add_labels(
                the_id=the_id, output_mode=output_mode, labels=labels)

    def set_label_names_action(output_mode, label_names):
        for split_name in split_to_ids: 
            split_name_to_compiler[split_name].set_label_names(
                output_mode=output_mode, label_names=label_names) 

    process_labels_with_labels_action(
        labels_objects=combined_yamls[RootKeys.keys.labels],
        labels_action=labels_action,
        set_label_names_action=set_label_names_action)

    #define the action that will get applied to new features
    def features_action(input_mode, the_id, features):
        split_names = id_to_split_names[the_id]
        for split_name in split_names:
            split_name_to_compiler[split_name].add_features(
                the_id=the_id, input_mode=input_mode, features=features)

    process_features_with_features_action(
        features_objects=combined_yamls[RootKeys.keys.features],
        features_action=features_action)

    #define the action that will be applied to new weights
    def weights_action(output_mode, the_id, weights):
        split_names = id_to_split_names[the_id]
        for split_name in split_names:
            split_name_to_compiler[split_name].add_weights(
                the_id=the_id, weights=weights, output_mode=output_mode)
    #TODO: implement weights action
  
    for compiler in split_name_to_compiler.values():
        compiler.finalize() 

    return split_name_to_compiler 

def process_features_with_features_action(features_objects,
                                          features_action): 
    for features_object in features_objects:
        FeaturesKeys.check_for_unsupported_keys_and_fill_in_defaults(
                     features_object)
        input_mode = features_object[FeaturesKeys.keys.input_mode_name] 

        features_format = features_object[FeaturesKeys.keys.features_format]
        features_opts = features_object[FeaturesKeys.keys.opts]

        features_iterator = get_features_iterator(features_format,
                                                  features_opts)

        for the_id, features in features_iterator:
            features_action(input_mode=input_mode,
                            the_id=the_id, features=features)
 

def get_features_iterator(features_format, features_opts):
    if features_format==FeaturesFormat.fasta:
        return fasta_iterator(features_opts)
    else:
        raise RuntimeError("Unsupported features format: "+features_format)


def fasta_iterator(features_opts):
    KeysObj = FeaturesFormatOptions_Fasta
    KeysObj.check_for_unsupported_keys_and_fill_in_defaults(features_opts)
    file_names = features_opts[KeysObj.keys.file_names] 
    progress_update = features_opts[KeysObj.keys.progress_update]
    for file_name in file_names: 
        print("on file",file_name)
        fasta_iterator = fp.FastaIterator(
                          file_handle=fp.get_file_handle(file_name),
                          progress_update=progress_update)
        for seq_id, seq in fasta_iterator:
            yield seq_id, av_util.seq_to_2d_image(seq) 


def process_labels_with_labels_action(labels_objects,
                                      labels_action,
                                      set_label_names_action): 
    print("Reading in labels")
    for labels_object in labels_objects:
        LabelsKeys.check_for_unsupported_keys_and_fill_in_defaults(
                    labels_object)
        output_mode = labels_object[LabelsKeys.keys.output_mode_name] 

        labels_file_name = labels_object[LabelsKeys.keys.file_name]
        file_with_subset_of_label_names =\
            labels_object[LabelsKeys.keys.file_with_subset_of_label_names]
        content_type=get_content_type_from_name(
                      labels_object[LabelsKeys.keys.content_type])
        subset_of_columns_to_use_options=\
          (None if file_with_subset_of_label_names is None
            else fp.SubsetOfColumnsToUseOptions(
                    column_names=fp.read_rows_into_arr(
                                    fp.get_file_handle(
                                    file_with_subset_of_label_names))))
        core_titled_mapping_action = fp.get_core_titled_mapping_action(
            subset_of_columns_to_use_options=\
                subset_of_columns_to_use_options,
                content_type=content_type,
                content_start_index=1,
                key_columns=labels_object[LabelsKeys.keys.key_columns])

        def action(inp, line_number):
            if (line_number==1):
                #If this is the first row, then pick out the list
                #of names relevant in the title
                label_names = core_titled_mapping_action(inp, line_number)
                set_label_names_action(output_mode=output_mode,
                                       label_names=label_names)
            else:
                #otherwise, pick out the labels
                the_id, labels = core_titled_mapping_action(inp, line_number)
                labels_action(output_mode=output_mode,
                              the_id=the_id,
                              labels=labels)
        fp.perform_action_on_each_line_of_file(
            file_handle=fp.get_file_handle(labels_file_name),
            action=action,
            transformation=fp.default_tab_seppd,
            progress_update=labels_object[LabelsKeys.keys.progress_update])
        

def get_id_to_split_names(split_object):
    """
        return:
        id_to_split_names
        split_to_ids
    """
    SplitKeys.check_for_unsupported_keys_and_fill_in_defaults(split_object)
    opts = split_object[SplitKeys.keys.opts]
    SplitOptsKeys.check_for_unsupported_keys_and_fill_in_defaults(opts)
    split_name_to_split_files = split_object[
                                SplitKeys.keys.split_name_to_split_files]
    id_to_split_names = {}
    split_to_ids = OrderedDict()
    for split_name in split_name_to_split_files:
        ids_in_split = fp.read_col_into_arr(
                        fp.get_file_handle(
                         split_name_to_split_files[split_name]), **opts)
        split_to_ids[split_name] = ids_in_split
        for the_id in ids_in_split:
            if the_id not in id_to_split_names:
                id_to_split_names[the_id] = []
            id_to_split_names[the_id].append(split_name)
    return id_to_split_names, split_to_ids
    

def get_content_type_from_name(content_type_name):
    if content_type_name not in ContentTypesLookup:
        raise RuntimeError("Unsupported content type: "+str(content_type_name))
    return ContentTypesLookup[content_type_name]

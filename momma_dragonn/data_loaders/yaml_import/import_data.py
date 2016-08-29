import sys
import os
import pdb 
from collections import namedtuple
from collections import OrderedDict
import numpy as np
import avutils.file_processing as fp


class DynamicEnum(object):
    """
        just a wrapper around a dictionary, so that the keys are
        accessible using the object attribute syntax rather
        than the dictionary syntax.
    """
    def __init__(self, *keys):
        self._vals_dict = OrderedDict()

    def add_key(self, key_name, val):
        setattr(self, key_name, val)
        self._vals_dict[key_name] = val

    def get_key(self, key_name):
        return self._vals_dict[key_name]

    def has_key(self, key_name):
        if key_name not in self._vals_dict:
            return False;
        return True;

    def get_keys(self):
        return self._vals_dict.keys()


class UNDEF(object):
    pass


class Key(object):
    def __init__(self, key_name_internal,
                       key_name_external=None, default=UNDEF):
        self.key_name_internal = key_name_internal
        if (key_name_external is None):
            key_name_external = key_name_internal
        self.key_name_external = key_name_external
        self.default = default


#I am keeping a different external and internal name
#for the flexibility of changing the external name in the future.
#the advantage of having a class like this rather than just
#using enums is being able to support methods like "fill_in_defaults_for_keys".
#I need the DynamicEnum class so that I can use the Keys class for
#different types of Keys; i.e. I don't
#know what the keys are going to be beforehand and I don't know
#how else to create an enum dynamically.
class Keys(object): 
    def __init__(self, *keys):
        #just a wrapper around a dictionary, for the
        #purpose of accessing the keys using the object
        #attribute syntax rather than the dictionary syntax.
        self.keys = DynamicEnum() 
        self.keys_defaults = DynamicEnum()
        for key in keys:
            self.add_key(key.key_name_internal, key.key_name_external,
                         key.default)

    def add_key(self, key_name_internal,
                      key_name_external, default_value=UNDEF):
        self.keys.add_key(key_name_internal, key_name_external)
        if (default_value != UNDEF):
            self.keys_defaults.add_key(key_name_internal, default_value)

    def check_for_unsupported_keys(self, a_dict):
        for a_key in a_dict:
            if self.keys.has_key(a_key)==False:
                raise RuntimeError("Unsupported key "+str(a_key)
                                   +"; supported keys are: "
                                   +str(self.keys.get_keys()))

    def fill_in_defaults_for_keys(self,
        a_dict, internal_names_of_keys_to_fill_defaults_for=None):
        if internal_names_of_keys_to_fill_defaults_for is None:
            internal_names_of_keys_to_fill_defaults_for = self.keys.get_keys()
        for a_key in internal_names_of_keys_to_fill_defaults_for:
            if a_key not in a_dict:
                if (self.keys_defaults.has_key(a_key)==False):
                    raise RuntimeError("Default for "+str(a_key)
                                       +" not present, and a value "
                                       +"was not provided")
                a_dict[a_key] = self.keys_defaults.getKey(a_key);
        return a_dict;

    def check_for_unsupported_keys_and_fill_in_defaults(self, a_dict):
        self.check_for_unsupported_keys(a_dict)
        self.fill_in_defaults_for_keys(a_dict);


ContentType=namedtuple('ContentType',['name','casting_function'])
ContentTypes=util.enum(integer=ContentType("int",int),
                       floating=ContentType("float",float),
                       string=ContentType("str",str))

ContentTypesLookup = dict((x.name,x.casting_function)
                          for x in ContentTypes.vals)

RootKeys=Keys(Key("features"), Key("labels"), Key("splits"), Key("weights"))

###
#Features Keys
###
FeaturesFormat=util.enum(rows_and_columns='rows_and_columns'
                         , fasta='fasta')
DefaultModeNames = util.enum(labels="default_output_mode_name",
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
#For files that have the format produced by getfasta bedtools; >key \n [fasta sequence] \n ...
FeatureSetYamlKeys_Fasta = Keys(Key("file_names"),
                                Key("progress_update",default=None))
SubsetOfColumnsToUseOptionsYamlKeys = Keys(
                                   Key("subset_of_columns_to_use_mode"),
                                   Key("file_with_column_names", default=None),
                                   Key("N", default=None))
###
#Label keys
###
LabelsKeys = Keys(Key("file_name")
                 , Key("content_type", default=ContentTypes.integer.name)
                 , Key("file_with_labels_to_use", default=None)
                 , Key("key_columns", default=[0])
                 , Key("output_mode_name", default=DefaultModeNames.labels))

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


def process_combined_yamls_for_input_datas(combined_yamls):
    #get the splits info 
    id_to_split_names, distinct_split_names = get_id_to_split_names(
                                          combined_yamls[RootKeys.keys.splits])
    output_mode_name_to_id_to_labels, output_mode_name_to_label_names =\
        get_id_to_labels(combined_yamls[RootKeys.keys.labels])
    output_mode_names = output_mode_name_to_id_to_labels.keys()
    

def get_id_to_split_names(split_object):
    """
        return:
        id_to_split_names
        distinct_split_names
    """
    SplitKeys.fill_in_defaults_for_keys(split_object)
    SplitKeys.check_for_unsupported_keys(split_object)
    opts = split_object[SplitKeys.keys.opts]
    SplitOptsKeys.fill_in_defaults_for_keys(opts)
    SplitOptsKeys.check_for_unsupported_keys(opts)
    split_name_to_split_file = split_object[
                                SplitKeys.keys.split_name_to_split_files]
    id_to_split_names = {}
    distinct_split_names = []
    for split_name in split_name_to_split_files:
        if split_name in distinct_split_names:
            raise RuntimeError("Repeated split_name: "+str(split_name))
        distinct_split_names.append(split_name)
        ids_in_split = fp.read_col_into_arr(
                        fp.get_file_handle(
                         split_name_to_split_file[split_name]), **opts)
        for the_id in ids_in_split:
            if the_id not in id_to_split_names:
                id_to_split_names[the_id] = []
            id_to_split_names[the_id].append(split_name)
    return id_to_split_names, distinct_split_names


def get_id_to_labels(labels_objects):
    """
        accepts a bunch of labels yaml objects, each of which corresponds to
            a different output mode.
        return: output_mode_name_to_id_to_labels,
                output_mode_name_to_label_names
    """
    output_mode_name_to_id_to_labels= OrderedDict()
    output_mode_name_to_label_names = OrderedDict()
    for labels_object in labels_objects:
        LabelsKeys.fill_in_defaults_for_keys(labels_object)
        LabelsKeys.check_for_unsupported_keys(labels_object)
        output_mode_name = labels_object[LabelsKeys.keys.output_mode_name] 
        titled_mapping_object = get_id_to_labels_single_label_set(
                                 labels_object)
        output_mode_name_to_id_to_labels[output_mode_name] =\
                                                titled_mapping_object.mapping
        output_mode_name_to_label_names[output_mode_name] =\
                                                titled_mapping_object.title_arr
    return output_mode_name_to_id_to_labels, output_mode_name_to_label_names


def get_id_to_labels_single_label_set(labels_object):
    """
        return: titled_mapping which has attributes:
            .mapping = id_to_labels dictionary
            .title_arr = label names.
    """
    file_with_labels_to_use =\
        labels_object[LabelsKeys.keys.file_with_labels_to_use]
    titled_mapping = fp.read_titled_mapping(
                      fp.get_file_handle(
                          labels_object[LabelsKeys.keys.file_name]),
                      content_type=get_content_type_from_name(
                       labels_object[LabelsKeys.keys.content_type]),
                      key_columns = labels_object[LabelsKeys.keys.key_columns],
                      subset_of_columns_to_use_options=\
                       (None if file_with_labels_to_use is None
                             else fp.SubsetOfColumnsToUseOptions(
                             columnNames=fp.readRowsIntoArr(
                                fp.getFileHandle(fileWithLabelsToUse))))
    return titledMapping;
    

def get_content_type_from_name(content_type_name):
    if content_type_name not in ContentTypesLookup:
        raise RuntimeError("Unsupported content type: "+str(content_type_name))
    return ContentTypesLookup[content_type_name]

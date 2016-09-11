from collections import OrderedDict
import yaml
from avutils import file_processing as fp
from avutils import util


def load_class_from_config(config, extra_kwargs={}):
    config = util.load_yaml_if_string(config)
    the_class = config['class']
    kwargs = config['kwargs']
    kwargs.add_all(extra_kwargs)
    return the_class(**kwargs)


def load_end_of_epoch_callbacks(config):
    config = util.load_yaml_if_string(config)
    end_of_epoch_callbacks = [load_class_from_config(callback_config)
                              for callback_config in config] 
    return end_of_epoch_callbacks


def load_end_of_training_callbacks(config, key_metric_name):
    config = util.load_yaml_if_string(config)
    end_of_training_callbacks = [
        load_class_from_config(callback_config,
            extra_kwargs={'key_metric_name': key_metric_name})
        for callback_config in config] 
    return end_of_training_callbacks 
    

def load_hyperparameter_configs(hyperparameter_configs_list):
    list_of_hyperparameter_settings = []   
    for hyperparameter_configs in\
        util.load_yaml_if_string(hyperparameter_configs_list):
        other_data_loaders = OrderedDict([
            (split_name, load_class_from_config(data_loader_config))
            for (split_name, data_loader_config) in
            hyperparameter_configs["other_data_loaders"].items()])
        model_creator = load_class_from_config(
                            hyperparameter_configs["model_creator"]) 
        model_trainer = load_class_from_config(
                            hyperparameter_configs["model_trainer"])
        list_of_hyperparameter_settings.append(
            {'other_data_loaders':other_data_loaders,
             'model_creator':model_creator,
             'model_trainer':model_trainer})
    return list_of_hyperparameter_settings
         

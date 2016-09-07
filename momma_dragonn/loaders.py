import yaml
from avutils import file_processing as fp

def load_class_from_config(config, extra_kwargs={}):
    config = load_file_if_string(config)
    the_class = config['class']
    kwargs = config['kwargs']
    kwargs.add_all(extra_kwargs)
    return the_class(**kwargs)

def load_callbacks(config):
    config = load_file_if_string(config)
    end_of_epoch_callbacks = [load_class_from_config(callback_config)
                    for callback_config in config["end_of_epoch_callbacks"]] 
    end_of_training_callbacks = [load_class_from_config(callback_config)
                    for callback_config in config["end_of_training_callbacks"]] 
    return end_of_epoch_callbacks, end_of_training_callbacks 

def load_file_if_string(config):
    if (isinstance(config, str)):
        config = yaml.load(fp.get_file_handle(config)) 
    return config
    
def load_hyperparameter_configs(hyperparameter_configs_list):
    list_of_hyperparameter_settings = []   
    for hyperparameter_configs in\
        load_file_if_string(hyperparameter_configs_list):
        other_data_loaders = [load_class_from_config(data_loader_config)
                              for data_loader_config in
                              hyperparameter_configs["other_data_loaders"]]
        model_creator = load_class_from_config(
                            hyperparameter_configs["model_creator"]) 
        model_trainer = load_class_from_config(
                            hyperparameter_configs["model_trainer"])
        list_of_hyperparameter_settings.append(
            {'other_data_loaders':other_data_loaders,
             'model_creator':model_creator,
             'model_trainer':model_trainer})
    return list_of_hyperparameter_settings
         

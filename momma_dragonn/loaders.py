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
    

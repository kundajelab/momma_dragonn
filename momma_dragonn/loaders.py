from collections import OrderedDict
import yaml
from avutils import file_processing as fp
from avutils import util
import re


def load_class_from_config(config, extra_kwargs={}, module_prefix=""):
    import momma_dragonn
    config = fp.load_yaml_if_string(config)
    path_to_class = module_prefix+config['class']
    print("Loading "+path_to_class)
    try:
        the_class = eval(path_to_class)
    except NameError:
        #parse out the beginning module and import before loading
        p = re.compile(r"^((.*)(\.))?([^.]*)$")
        m = p.search(path_to_class)
        module,the_class = m.group(2),m.group(4)
        if (module is not None):
            exec("import "+module)
        the_class = eval(path_to_class)
    kwargs = config['kwargs']
    kwargs.update(extra_kwargs)
    return the_class(**kwargs)


def load_data_loader(config, extra_kwargs={}):
    return load_class_from_config(config=config, extra_kwargs=extra_kwargs,
                module_prefix="momma_dragonn.data_loaders.")


def load_stopping_criterion(config, extra_kwargs={}):
    return load_class_from_config(config=config, extra_kwargs=extra_kwargs,
                module_prefix="momma_dragonn.stopping_criteria.")


def load_model_evaluator(config, extra_kwargs={}):
    return load_class_from_config(config=config, extra_kwargs=extra_kwargs,
                module_prefix="momma_dragonn.model_evaluators.")


def load_end_of_epoch_callbacks(config):
    config = fp.load_yaml_if_string(config)
    end_of_epoch_callbacks = [
        load_class_from_config(config=callback_config,
            module_prefix="momma_dragonn.end_of_epoch_callbacks.")
        for callback_config in config] 
    return end_of_epoch_callbacks


def load_end_of_training_callbacks(config, key_metric_name, larger_is_better):
    config = fp.load_yaml_if_string(config)
    end_of_training_callbacks = [
        load_class_from_config(config=callback_config,
            extra_kwargs={'key_metric_name': key_metric_name,
                          'larger_is_better': larger_is_better},
            module_prefix="momma_dragonn.end_of_training_callbacks.")
        for callback_config in config] 
    return end_of_training_callbacks 
    

def load_hyperparameter_configs_list(hyperparameter_configs_list):
    list_of_hyperparameter_settings = []   
    for hyperparameter_configs in\
        fp.load_yaml_if_string(hyperparameter_configs_list):
        other_data_loaders = OrderedDict([
            (split_name, load_data_loader(data_loader_config))
            for (split_name, data_loader_config) in
            hyperparameter_configs["other_data_loaders"].items()])
        model_creator = load_class_from_config(
                            config=hyperparameter_configs["model_creator"],
                            module_prefix="momma_dragonn.model_creators.") 
        model_trainer = load_class_from_config(
                            config=hyperparameter_configs["model_trainer"],
                            module_prefix="momma_dragonn.model_trainers.")
        list_of_hyperparameter_settings.append(
            {'other_data_loaders':other_data_loaders,
             'model_creator':model_creator,
             'model_trainer':model_trainer})
    return list_of_hyperparameter_settings 

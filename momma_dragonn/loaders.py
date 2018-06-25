from collections import OrderedDict
from avutils import file_processing as fp
from avutils import util
import re


def load_class_from_config(config, extra_kwargs={}, module_prefix=""):
    import momma_dragonn
    config = fp.load_yaml_if_string(config)
    if ('modules_to_load' in config):
        for a_module in config['modules_to_load']:
            exec("import "+a_module)
    if ('func' in config):
        return eval(config['func'])
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
    parsed_kwargs = {}
    #replace any recursive kwargs as necessary
    for key,val in kwargs.items():
        if (isinstance(val,dict) and (len(val.keys())==3) and ('class' in val)
                and ('kwargs' in val) and ('autoload' in val)
                and (val['autoload']==True)): 
                parsed_kwargs[key] = load_class_from_config(config=val)
        else:
            parsed_kwargs[key] = val
    parsed_kwargs.update(extra_kwargs)
    return the_class(**parsed_kwargs)


def load_data_loader(config, extra_kwargs={}):
    return load_class_from_config(config=config, extra_kwargs=extra_kwargs,
                module_prefix="momma_dragonn.data_loaders.")


def load_stopping_criterion(config, extra_kwargs={}):
    return load_class_from_config(config=config, extra_kwargs=extra_kwargs,
                module_prefix="momma_dragonn.stopping_criteria.")


def load_model_evaluator(config, extra_kwargs={}):
    return load_class_from_config(config=config, extra_kwargs=extra_kwargs,
                module_prefix="momma_dragonn.model_evaluators.")


def load_epoch_callbacks(config):
    config = fp.load_yaml_if_string(config)
    epoch_callbacks = [
        load_class_from_config(config=callback_config,
            module_prefix="momma_dragonn.epoch_callbacks.")
        for callback_config in config] 
    return epoch_callbacks


def load_end_of_training_callbacks(config, key_metric_name, larger_is_better):
    config = fp.load_yaml_if_string(config)
    end_of_training_callbacks = [
        load_class_from_config(config=callback_config,
            extra_kwargs={'key_metric_name': key_metric_name,
                          'larger_is_better': larger_is_better},
            module_prefix="momma_dragonn.end_of_training_callbacks.")
        for callback_config in config] 
    return end_of_training_callbacks 
    
def load_hyperparameter_block(hyperparameter_configs):
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
    if ('message' in hyperparameter_configs):
        message = hyperparameter_configs['message']
    else:
        message = ""
    return {'other_data_loaders':other_data_loaders,
             'model_creator':model_creator,
             'model_trainer':model_trainer,
             'message': message}


def load_class_from_config(config):
    the_class = config['class']
    kwargs = config['kwargs']
    return the_class(**kwargs)

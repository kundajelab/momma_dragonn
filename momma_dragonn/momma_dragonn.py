
def load_class_from_config(config, extra_kwargs):
    the_class = config['class']
    kwargs = config['kwargs']
    kwargs.add_all(extra_kwargs)
    return the_class(**kwargs)

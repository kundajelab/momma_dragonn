
class AbstractModelCreator(object):
    def __init__(self, config):
        self.config = config

    def get_model(self):
        raise NotImplementedError()

    def get_config(self):
        return self.config

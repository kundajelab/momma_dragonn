from .core import AbstractModelCreator
from avutils.dynamic_enum import Key, Keys

FlexibleKerasKeys = Keys(Key("layers"),
                         Key("optimizer"),
                         Key("objective"))

class FlexibleKeras(AbstractModelCreator):
    def __init__(self, config):
        self.config = config

    def get_model(self):
        uncompiled_model = self._get_uncompiled_model() 
        compiled_model = self._compile_model(uncompiled_model)
        return compiled_model

    def _get_uncompiled_model(self):
        raise NotImplementedError()

    def _compile_model(self, model):
        raise NotImplementedError()

    def get_config(self):
        return self.config

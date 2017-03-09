from .core import AbstractModelCreator
from avutils.dynamic_enum import Key, Keys
from collections import OrderedDict
from ..model_wrappers import keras_model_wrappers
from momma_dragonn.loaders import load_class_from_config
#DO NOT import keras at the top level; the random
#seed needs to be set BEFORE keras is imported


class KerasModelFromFunc(AbstractModelCreator):

    def __init__(self, func, model_wrapper_class):
        self.func = func
        self._last_model_created = None
        self.model_wrapper_class = model_wrapper_class

    def get_model_wrapper(self, seed):
        #seed is ignored
        self.create_model()
        model_wrapper = self.model_wrapper_class()
        model_wrapper.set_model(self._last_model_created) 
        return model_wrapper

    def create_model(self):
        self._last_model_created = self.func()
        assert self._last_model_created is not None, "func() returned None"

    def get_jsonable_object(self):
        if (self._last_model_created is None):
            print("jsonable object requested but model not created; creating")
            self.create_model()
        return OrderedDict(['model_json', self._last_model_created.to_json()])


class FlexibleKeras(AbstractModelCreator):

    def _get_uncompiled_model(self):
        raise NotImplementedError()

    def _parse_loss(self):
        raise NotImplementedError()

    def get_model(self, seed):
        print("Preparing uncompiled model")
        model = self._get_uncompiled_model(seed) 
        print("Compiling model")
        self._compile_model(model)
        print("Done compiling model")
        return model

    def _compile_model(self, model):
        optimizer = load_class_from_config(self.optimizer_config)
        model.compile(optimizer=optimizer, loss=self._parse_loss()) 


class FlexibleKerasGraph(FlexibleKeras):

    def __init__(self, inputs_config, nodes_config, outputs_config,
                       optimizer_config, loss_dictionary):
        self.inputs_config = inputs_config
        self.nodes_config = nodes_config
        self.outputs_config = outputs_config
        self.optimizer_config = optimizer_config
        self.loss_dictionary = loss_dictionary

    def get_model_wrapper(self, seed):
        model_wrapper = keras_model_wrappers.KerasGraphModelWrapper()
        model_wrapper.set_model(self.get_model(seed=seed))
        return model_wrapper

    def get_jsonable_object(self):
        return OrderedDict([
                ('inputs_config', self.inputs_config),
                ('nodes_config', self.nodes_config),
                ('outputs_config', self.outputs_config),
                ('optimizer_config', self.optimizer_config),
                ('loss_dictionary', self.loss_dictionary)])

    def _get_uncompiled_model(self):
        from keras.legacy.models import Graph 
        graph = Graph()
        self._add_inputs(graph) 
        self._add_nodes(graph)
        self._add_outputs(graph)
        return graph

    def _add_inputs(self, graph):
        for input_config in self.inputs_config:
           graph.add_input(**input_config) 

    def _add_nodes(self, graph):
        from momma_dragonn.loaders import load_class_from_config
        for node_config in self.nodes_config:
            the_class = node_config["class"]
            the_kwargs = node_config["kwargs"]
            layer = load_class_from_config(
                        {'class': the_class,
                         'kwargs': the_kwargs},
                         extra_kwargs={})

            add_node_kwargs = {}
            for a_key in node_config:
                if a_key not in ["class", "kwargs", "input_name"]:
                    add_node_kwargs[a_key] = node_config[a_key] 

            if (isinstance(node_config["input_name"],list)):
                add_node_kwargs['inputs'] = node_config["input_name"]
            elif (isinstance(node_config["input_name"], str)):
                add_node_kwargs['input'] = node_config["input_name"]
            else:
                raise RuntimeError("Unsupported type for input_name: "
                                   +str(node_config["input_name"]))

            graph.add_node(
                layer,
                **add_node_kwargs)

    def _add_outputs(self, graph):
        for output_config in self.outputs_config:
            graph.add_output(**output_config) 

    def _parse_loss(self):
        parsed_loss_dictionary =\
            dict((key, (val if isinstance(val,dict)==False else
                        load_class_from_config(val))) for
                        (key,val) in self.loss_dictionary.items())
        return parsed_loss_dictionary


class FlexibleKerasSequential(FlexibleKeras):

    def __init__(self, layers_config, optimizer_config, loss):
        self.layers_config = layers_config
        self.optimizer_config = optimizer_config
        self.loss = loss

    def get_model_wrapper(self, seed):
        model_wrapper = keras_model_wrappers.KerasModelWrapper()
        model_wrapper.set_model(self.get_model(seed=seed))
        return model_wrapper 

    def get_jsonable_object(self):
        return OrderedDict([
                ('layers_config', self.layers_config),
                ('optimizer_config', self.optimizer_config),
                ('loss', self.loss)])

    def _parse_loss(self):
        if (isinstance(self.loss, str)):
            return self.loss
        else:
            return load_class_from_config(self.loss) 

    def _get_uncompiled_model(self, seed):
        #it is important that keras is only imported here so that
        #the random seed can be set by the model trainer BEFORE the import
        import numpy as np
        np.random.seed(seed)
        import keras
        from keras.models import Sequential
        model = Sequential()
        for layer_config in self.layers_config:
            model.add(load_class_from_config(layer_config)) 
        return model

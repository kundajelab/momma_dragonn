from .core import AbstractModelCreator
from avutils.dynamic_enum import Key, Keys
import keras
from momma_dragonn.loaders import loaders
from collections import OrderedDict
import keras


class FlexibleKerasGraph(AbstractModelCreator):
    def __init__(self, input_layers_config, nodes_config,
                       optimizer_config, loss_dictionary):
        self.input_layers_config = input_layers_config
        self.nodes_config = nodes_config
        self.optimizer_config = optimizer_config
        self.loss_dictionary = loss_dictionary

    def get_jsonable_object(self):
        return OrderedDict([
                ('input_layers_config', self.input_layers_config),
                ('nodes_config', self.nodes_config),
                ('optimizer_config', self.optimizer_config),
                ('loss_dictionary', self.loss_dictionary)])

    def get_model(self):
        model = self._get_uncompiled_model() 
        self._compile_model(model)
        return model

    def _get_uncompiled_model(self):
        from keras.model import Graph 
        graph = Graph()
        self._add_input_layers(graph) 
        self._add_nodes(graph)
        return graph

    def _add_input_layers(self, graph):
        for input_layer_config in self.input_layers_config:
           graph.add_input(name=input_layers_config["name"],
                           input_shape=input_layers_config["input_shape"]) 

    def _add_nodes(self, graph):
        for node_config in self.nodes_config:
            class_and_kwargs = node_config["class_and_kwargs"]
            layer = load_class_from_config(class_and_kwargs, extra_kwargs={}):
            graph.add_node(layer, name=node_config["name"],
                           input=node_config["input_name"])

    def _compile_model(self, graph):
        optimizer = load_class_from_config(self.optimizer_config)
        graph.compile(optimizer=optimizer, loss=self.loss_dictionary) 

    def get_config(self):
        return self.config

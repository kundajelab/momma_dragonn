from .core import AbstractModelCreator
from avutils.dynamic_enum import Key, Keys
import keras
from momma_dragonn.loaders import loaders
from collections import OrderedDict
import keras


class FlexibleKerasGraph(AbstractModelCreator):
    def __init__(self, inputs_config, nodes_config, outputs_config,
                       optimizer_config, loss_dictionary):
        self.inputs_config = inputs_config
        self.nodes_config = nodes_config
        self.outputs_config = outputs_config
        self.optimizer_config = optimizer_config
        self.loss_dictionary = loss_dictionary

    def get_jsonable_object(self):
        return OrderedDict([
                ('inputs_config', self.inputs_config),
                ('nodes_config', self.nodes_config),
                ('outputs_config', self.outputs_config),
                ('optimizer_config', self.optimizer_config),
                ('loss_dictionary', self.loss_dictionary)])

    def get_model(self):
        model = self._get_uncompiled_model() 
        self._compile_model(model)
        return model

    def _get_uncompiled_model(self):
        from keras.model import Graph 
        graph = Graph()
        self._add_inputs(graph) 
        self._add_nodes(graph)
        self._add_outputs(graph)
        return graph

    def _add_input_layers(self, graph):
        for input_config in self.inputs_config:
           graph.add_input(**input_config) 

    def _add_nodes(self, graph):
        for node_config in self.nodes_config:
            the_class = node_config["class"]
            the_kwargs = node_config["kwargs"]
            layer = load_class_from_config(
                        {'class': the_class,
                         'the_kwargs': the_kwargs},
                         extra_kwargs={})
            graph.add_node(layer, name=node_config["name"],
                           input=node_config["input_name"])

    def _add_outputs(self, graph):
        for output_config in self.outputs_config:
            graph.add_output(**output_config) 

    def _compile_model(self, graph):
        optimizer = load_class_from_config(self.optimizer_config)
        graph.compile(optimizer=optimizer, loss=self.loss_dictionary) 

    def get_config(self):
        return self.config

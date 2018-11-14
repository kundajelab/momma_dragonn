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
        return OrderedDict([('model_json',
                              self._last_model_created.to_json())])


class KerasModelFromSavedFile(AbstractModelCreator):

    def __init__(self, weight_file, json_file, optimizer_config, loss, metrics=[]):
        self.weight_file = weight_file
        self.json_file = json_file
        self.optimizer_config = optimizer_config
        self.loss = loss
        self.metrics = metrics

    def get_model(self, seed):
        print("Preparing uncompiled model")
        model = self._get_uncompiled_model(seed) 
        print("Compiling model")
        self._compile_model(model)
        print("Done compiling model")
        return model

    def _compile_model(self, model):
        optimizer = load_class_from_config(self.optimizer_config)
        if (isinstance(self.loss, str)):
            loss = self.loss
        else:
            loss = load_class_from_config(self.loss) 
        model.compile(optimizer=optimizer, loss=loss, metrics=self.metrics) 
    

    def get_model_wrapper(self, seed):
        #create the model 
        if (self.json_file is not None):
            from keras.models import model_from_json 
            model = model_from_json(open(self.json_file).read())
        else:
            from keras.models import load_model 
            model = load_model(self.weight_file)

        if type(model).__name__ == "Sequential":
            model_wrapper  = keras_model_wrappers.KerasModelWrapper() 
        elif type(model).__name__ == "Model":
            output_names = model.output_names 
            model_wrapper  =\
                keras_model_wrappers.KerasFunctionalModelWrapper(
                    output_names=output_names) 
        else:
            raise RuntimeError("Unrecognized model name: "
                               +type(model).__name__)

        model_wrapper.set_model(self.get_model(seed=seed))
        return model_wrapper

    def get_jsonable_object(self):
        return OrderedDict([('weight_file', self.weight_file),
                            ('json_file', self.json_file),
                            ('optimizer_config', self.optimizer_config),
                            ('loss', self.loss)])

    def _get_uncompiled_model(self, seed):
        #it is important that keras is only imported here so that
        #the random seed can be set by the model trainer BEFORE the import
        import numpy as np
        np.random.seed(seed)
        import keras
        if (self.json_file is not None):
            from keras.models import model_from_json
            model = model_from_json(open(self.json_file).read())
            model.load_weights(self.weight_file)
        else:
            from keras.models import load_model 
            model = load_model(self.weight_file)
            
        return model


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
        print(model.summary())
        return model

    def _compile_model(self, model):
        optimizer = load_class_from_config(self.optimizer_config)
        model.compile(optimizer=optimizer, loss=self._parse_loss(),metrics=self.metrics) 


class ParseLossDictionaryMixin(object):
    def _parse_loss(self):
        parsed_loss_dictionary =\
            dict((key, (val if isinstance(val,dict)==False else
                        load_class_from_config(val))) for
                        (key,val) in self.loss_dictionary.items())
        return parsed_loss_dictionary


class FlexibleKerasGraph(FlexibleKeras, ParseLossDictionaryMixin):

    def __init__(self, inputs_config, nodes_config, outputs_config,
                       optimizer_config, loss_dictionary,metrics=[]):
        self.inputs_config = inputs_config
        self.nodes_config = nodes_config
        self.outputs_config = outputs_config
        self.optimizer_config = optimizer_config
        self.loss_dictionary = loss_dictionary
        self.metrics=[] 


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

    def _get_uncompiled_model(self, seed):
        #it is important that keras is only imported here so that
        #the random seed can be set by the model trainer BEFORE the import
        import numpy as np
        np.random.seed(seed)
        import keras
        from keras.models import Graph
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


class FlexibleKerasFunctional(ParseLossDictionaryMixin, FlexibleKeras):

    def __init__(self, input_names,
                       nodes_config,
                       output_names,
                       optimizer_config,
                       loss_dictionary,
                       shared_layers_config={},
                       metrics=[]):
        self.input_names = input_names
        self.nodes_config = nodes_config
        self.output_names = output_names
        self.optimizer_config = optimizer_config
        self.loss_dictionary = loss_dictionary
        self.shared_layers_config = shared_layers_config
        self.metrics=metrics 

    def get_model_wrapper(self, seed):
        model_wrapper = keras_model_wrappers.KerasFunctionalModelWrapper(
                         self.output_names)
        model_wrapper.set_model(self.get_model(seed=seed))
        return model_wrapper

    def get_jsonable_object(self):
        return OrderedDict([
                ('input_names', self.input_names),
                ('shared_layers_config', self.shared_layers_config),
                ('nodes_config', self.nodes_config),
                ('output_names', self.output_names),
                ('optimizer_config', self.optimizer_config),
                ('loss_dictionary', self.loss_dictionary)])

    def _get_uncompiled_model(self, seed):
        #it is important that keras is only imported here so that
        #the random seed can be set by the model trainer BEFORE the import
        import numpy as np
        np.random.seed(seed)
        import keras

        #first load all the shared layers, indexed by their name
        #every shared layer must be given a name so we can refer to
        #it later on!
        shared_layers = self._get_shared_layers()
       
        name_to_tensor = {} 
        for node_name, node_config in self.nodes_config.items():
            assert_attributes_in_config(node_config, ["layer"]) 

            #if layer is not an input layer, collect all the input tensors
            if (isinstance(node_config['layer'],str)
                or node_config['layer']['class'].endswith(".Input")==False):
                assert_attributes_in_config(node_config, ["input_node_names"])
                input_node_names = node_config['input_node_names']
                if (isinstance(input_node_names, list)):
                    input_tensors = []
                    for input_node_name in input_node_names:
                        assert input_node_name in name_to_tensor,\
                         (input_node_name
                          +" hasn't been declared already; declared "
                          +"node names are: "+str(name_to_tensor.keys()))
                        input_tensors.append(name_to_tensor[input_node_name])
                elif (isinstance(input_node_names, str)):
                    assert input_node_names in name_to_tensor,\
                     (input_node_names+" for "+str(node_config)+" hasn't been"
                      +" declared already; declared "
                      +"node names are: "+str(name_to_tensor.keys()))
                    input_tensors = name_to_tensor[input_node_names]
                else:
                    raise RuntimeError("Unsupported type for input_node_names: "
                          +str(type(input_node_names)))

            #now load the layer.
            layer_config = node_config['layer']
            #if 'layer_config' is just a string, it should refer to the
            #name of a shared layer
            if (isinstance(layer_config, str)): 
                assert layer_config in shared_layers,\
                (layer_config+" not in shared_layers; shared_layers are: "
                 +str(shared_layers.keys()))
                node_tensor = shared_layers[layer_config](input_tensors)
            #if it's a dictionary, can either be a layer declaration or
            #a merge function
            elif (isinstance(layer_config, dict)):
                assert_attributes_in_config(layer_config, ['class', 'kwargs'])
                assert 'name' not in layer_config['kwargs'],\
                 ("Don't declare 'name' within the kwargs; "
                  +" will use the dictionary key for that. At: "
                  +str(self.shared_layers_config))
                layer_config_class = layer_config['class']
                #when it's a merge function, we need to pass in input_tensors
                #as the inputs argument
                if layer_config_class.endswith('.merge'):
                    node_tensor = load_class_from_config(
                                   layer_config,
                                   extra_kwargs={
                                    'name': node_name, 
                                    'inputs': input_tensors})
                #otherwise, we call the layer object on the input
                #tensor after it has been instantiated 
                elif (layer_config_class.endswith('.Input')):
                    node_tensor = load_class_from_config(layer_config,
                                    extra_kwargs={'name': node_name})
                else:
                    node_tensor = (load_class_from_config(layer_config,
                                    extra_kwargs={'name': node_name})
                                    (input_tensors))
            else:
                raise RuntimeError("Unsupported type for node_layer_config "
                                   +str(type(layer_config)))
            #record the node tensor according to node_name
            name_to_tensor[node_name] = node_tensor

        for name in self.input_names+self.output_names:
            if name not in name_to_tensor:
                raise RuntimeError("No node with name: "+name
                  +" declared. Node names are: "+str(name_to_tensor.keys()))
         
        from keras.models import Model
        model = Model(input=[name_to_tensor[x] for x in self.input_names],
                      output=[name_to_tensor[x] for x in self.output_names])  
        return model

    def _get_shared_layers(self):
        shared_layers = {}
        for name,shared_layer_config in self.shared_layers_config.items():
            assert_attributes_in_config(shared_layer_config,
                                        ['class', 'kwargs'])
            assert 'name' not in shared_layer_config['kwargs'],\
             ("Don't declare 'name' within the kwargs; will use the dictionary"
              +" key for that. At: "+str(shared_layer_config))
            shared_layer = load_class_from_config(
                            shared_layer_config,
                            extra_kwargs={'name': name})
            if (name in shared_layers):
                raise RuntimeError("Duplicated shared layer: "+str(name))
            shared_layers[name] = shared_layer
        return shared_layers


def assert_attributes_in_config(config, attributes):
    for attribute in attributes:
        assert attribute in config,\
         "missing attribute "+str(attribute)+" from "+str(config)


class FlexibleKerasSequential(FlexibleKeras):

    def __init__(self, layers_config,
                       optimizer_config,
                       loss,
                       pretrained_model_config=None,
                       metrics=[]):
        self.layers_config = layers_config
        self.optimizer_config = optimizer_config
        self.loss = loss
        self.pretrained_model_config = pretrained_model_config
        self.metrics=metrics

    def get_model_wrapper(self, seed):
        model_wrapper = keras_model_wrappers.KerasModelWrapper()
        model_wrapper.set_model(self.get_model(seed=seed))
        return model_wrapper 

    def get_jsonable_object(self):
        return OrderedDict([
                ('pretrained_model_config', self.pretrained_model_config),
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

        if (self.pretrained_model_config is not None):
            pretrained_model_weights = self.pretrained_model_config["weight_file"]
            pretrained_model_json = self.pretrained_model_config["json_file"]
            last_layer_to_take =\
                self.pretrained_model_config["last_layer_to_take"]
            last_layer_to_fix =\
                (self.pretrained_model_config["last_layer_to_fix"]
                 if "last_layer_to_fix" in self.pretrained_model_config
                 else None)
            if (pretrained_model_json is not None):
                from keras.models import model_from_json
                pre_model =\
                    model_from_json(open(pretrained_model_json).read())
                pre_model.load_weights(pretrained_model_weights)
            else:
                from keras.models import load_model 
                pre_model = load_model(pretrained_model_weights)
            for idx,a_layer in enumerate(
                                pre_model.layers[:last_layer_to_take]):
                if (last_layer_to_fix is not None):
                    if idx <= ((len(pre_model.layers)+last_layer_to_fix)
                               if last_layer_to_fix else last_layer_to_take): 
                        a_layer.trainable=False
                model.add(a_layer) 

        for layer_config in self.layers_config:
            model.add(load_class_from_config(layer_config)) 
        return model

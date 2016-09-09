from avutils import file_processing as fp
from avutils import util
from collections import OrderedDict

class AbstractModelWrapper(object):

    def __init__(self, model):
        self.model
        #randomly generated string to use as an id 
        self.random_string = util.get_random_string(5)

    def get_model(self):
        return self.model

    def predict(self, X):
        raise NotImplementedError()

    def create_files_to_save(self, directory, prefix):
        raise NotImplementedError()

    def prefix_to_last_saved_files(self, prefix, new_directory=None):
        raise NotImplementedError()

    def get_last_saved_files_config(self):
        raise NotImplementedError()


class KerasGraphModelWrapper(AbstractModelWrapper):

    def __init__(self, **kwargs):
        super(KerasGraphModelWrapper, self).__init__(**kwargs)
        self.last_saved_files_config = {}

    def predict(self, X, batch_size):
        return self.model.predict(X,batch_size=batch_size)

    def generate_file_names(self, directory, prefix):
        file_path_prefix = directory+"/"+prefix+"_" 
        weights_file = file_path_prefix+"_modelWeights.h5"
        yaml_file = file_path_prefix+"_modelYaml.yaml"
        return weights_file, yaml_file

    def create_files_to_save(self, directory, prefix):
        weights_file, yaml_file = self.generate_file_names(
                                        directory, prefix)
        model.save_weights(weights_file,overwrite=True)
        fp.write_to_file(yaml_file, model.to_yaml())
        self.last_saved_files_config =\
            OrderedDict([('weights_file', weights_file),
                         ('yaml_file', yaml_file),
                         ('directory', directory),
                         ('prefix', prefix)])
        
    def prefix_to_last_saved_files(self, prefix, new_directory=None):
        if new_directory is None:
            new_directory = self.last_saved_files_config['directory'] 
        new_prefix = prefix+"_"+self.last_saved_files_config['prefix']
        old_weights = self.last_saved_files_config['weights_file']
        old_yaml = self.last_saved_files_config['yaml_file']
        new_weights, new_yaml =\
            self.generate_file_names(new_directory, new_prefix)
        fp.rename_files([(old_weights, new_weights), (old_yaml, new_yaml)])
        

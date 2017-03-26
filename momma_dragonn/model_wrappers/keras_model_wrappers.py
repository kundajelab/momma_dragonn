from avutils import file_processing as fp
from avutils import util
from collections import OrderedDict
from .core import AbstractModelWrapper


class KerasModelWrapper(AbstractModelWrapper):

    def __init__(self, **kwargs):
        super(KerasModelWrapper, self).__init__(**kwargs)
        self.last_saved_files_config = {}

    def predict(self, X, batch_size):
        return self.model.predict_proba(X,batch_size=batch_size)

    def generate_file_names(self, directory, prefix):
        file_path_prefix = directory+"/"+prefix 
        weights_file = file_path_prefix+"_modelWeights.h5"
        #yaml_file = file_path_prefix+"_modelYaml.yaml"
        #return weights_file, yaml_file
        json_file = file_path_prefix+"_modelJson.json"
        return weights_file, json_file

    def create_files_to_save(self, directory, prefix, update_last_saved):
        util.create_dir_if_not_exists(directory)   
        weights_file, json_file = self.generate_file_names(
                                        directory, prefix)
        self.model.save_weights(weights_file,overwrite=True)
        fp.write_to_file(json_file, self.model.to_json())
        if (update_last_saved): #for tracking best validation set
            self.last_saved_files_config =\
                OrderedDict([('weights_file', weights_file),
                             ('json_file', json_file),
                             ('directory', directory),
                             ('prefix', prefix)])
        
    def prefix_to_last_saved_files(self, prefix, new_directory=None):
        if new_directory is None:
            if 'directory' not in self.last_saved_files_config:
                raise RuntimeError("saved_files config is uninitialized; "
            "program must have terminated abnormally; look above for an error")
            else:
                new_directory = self.last_saved_files_config['directory'] 
        util.create_dir_if_not_exists(new_directory)   
        new_prefix = prefix+"_"+self.last_saved_files_config['prefix']
        old_weights = self.last_saved_files_config['weights_file']
        old_json = self.last_saved_files_config['json_file']
        new_weights, new_json =\
            self.generate_file_names(new_directory, new_prefix)
        fp.rename_files([(old_weights, new_weights), (old_json, new_json)])

        self.last_saved_files_config['directory'] = new_directory
        self.last_saved_files_config['weights_file'] = new_weights
        self.last_saved_files_config['json_file'] = new_json


class KerasGraphModelWrapper(KerasModelWrapper):

    def predict(self, X, batch_size):
        return self.model.predict(X,batch_size=batch_size)


class KerasFunctionalModelWrapper(KerasModelWrapper):

    def __init__(self, output_names, **kwargs):
        self.output_names = output_names
        super(KerasFunctionalModelWrapper, self).__init__(**kwargs)

    def predict(self, X, batch_size):
        predictions = self.model.predict(X,batch_size=batch_size)
        if (len(predictions) > len(self.output_names)
            and len(self.output_names)==1):
            predictions = [predictions] #got unlisted in background
        return dict(zip(self.output_names,predictions))

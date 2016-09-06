from avutils import file_processing as fp

class AbstractModelWrapper(object):

    def __init__(self, model):
        self.model

    def get_model(self):
        return self.model

    def predict(self, X):
        raise NotImplementedError()

    def create_files_to_save(self, directory):
        raise NotImplementedError()


class KerasGraphModelWrapper(AbstractModelWrapper):

    def predict(self, X, batch_size):
        return self.model.predict(X,batch_size=batch_size)

    def create_files_to_save(self, directory, name_prefix):
        file_path_prefix = directory+"/"+name_prefix+"_" 
        model.save_weights(file_path_prefix+"_modelWeights.h5",overwrite=True)
        fp.write_to_file(file_path_prefix+"_modelYaml.yaml", model.to_yaml())
        

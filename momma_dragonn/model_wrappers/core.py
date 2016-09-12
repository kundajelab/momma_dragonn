from avutils import util

class AbstractModelWrapper(object):

    def __init__(self, model):
        self.model = model
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

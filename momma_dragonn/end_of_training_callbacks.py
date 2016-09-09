from avutils import util

class AbstractEndOfTrainingCallback(object):

    def __call__(self, **kwargs):
        raise NotImplementedError()


class WriteToDbCallback(AbstractEndOfTrainingCallback):

    def __init__(self, db_path, key_metric_name, **kwargs):
        self.key_metric_name = key_metric_name 
        self.db_path = ( #put the perf metric in the db name
            util.get_file_name_parts(db_path) 
                .get_transformed_file_path(transformation:
                 lambda x: x+"_perf-metric-"+str(self.key_metric_name)))

    def __call__(self, performance_history, model_wrapper, training_metadata):
    


class PerformanceHistory(object):

    def __init__(self):
        self.training_key_metric_history = []
        self.validation_key_metric_history = []

    def update_training_key_metric(self, key_metric):
        self.training_key_metric_history.append(key_metric)
          
    def update_validation_key_metric(self, key_metric):
        self.validation_key_metric_history.append(key_metric)

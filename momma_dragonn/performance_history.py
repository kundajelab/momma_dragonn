
class PerformanceHistory(object):

    def __init__(self):
        self._train_key_metric_history = []
        self._valid_key_metric_history = []
        self._best_valid_epoch_perf_info = None

    def update_train_key_metric(self, key_metric):
        self._train_key_metric_history.append(key_metric)
          
    def update_valid_key_metric(self, key_metric):
        self._valid_key_metric_history.append(key_metric)

    def update_best_valid_epoch_perf_info(self, perf_info):
        self._best_valid_epoch_perf_info = perf_info

    def get_best_valid_epoch_perf_info(self):
        return self._best_valid_epoch_perf_info

    def get_train_key_metric_history(self):
        return self._train_key_metric_history

    def get_valid_key_metric_history(self):
        return self._valid_key_metric_history

from collections import OrderedDict


class PerfInfo(object):
    def __init__(self, epoch, valid_key_metric, train_key_metric,
                       valid_all_stats):
        self.epoch = epoch
        self.valid_key_metric = valid_key_metric
        self.train_key_metric = train_key_metric
        self.valid_all_stats = valid_all_stats

    def get_jsonable_object(self):
        return OrderedDict([
            ('epoch', self.epoch),
            ('valid_key_metric', self.valid_key_metric),
            ('train_key_metric', self.train_key_metric),
            ('valid_all_stats', self.valid_all_stats)])


class PerformanceHistory(object):

    def __init__(self):
        self._train_key_metric_history = []
        self._valid_key_metric_history = []
        self._best_valid_epoch_perf_info = None

    def epoch_update(self, train_key_metric, valid_key_metric):
        self._train_key_metric_history.append(train_key_metric)
        self._valid_key_metric_history.append(valid_key_metric)

    def update_best_valid_epoch_perf_info(self, **kwargs):
        self._best_valid_epoch_perf_info = PerfInfo(**kwargs)

    def get_best_valid_epoch_perf_info(self):
        return self._best_valid_epoch_perf_info

    def get_train_key_metric_history(self):
        return self._train_key_metric_history

    def get_valid_key_metric_history(self):
        return self._valid_key_metric_history

    def get_jsonable_object(self):
        returnable =  OrderedDict([
            ('train_key_metric_history', self._train_key_metric_history),
            ('valid_key_metric_history', self._valid_key_metric_history)])
        if self._best_valid_epoch_perf_info is not None:
            returnable.update({'best_valid_epoch_perf_info': self._best_valid_epoch_perf_info.get_jsonable_object()})
        return returnable

    def __len__(self):
        return len(self._train_key_metric_history)

import avutils.util as util
import numpy as np

class AbstractStoppingCriterion(object):

    def __init__(self, max_epochs):
        self.max_epochs = max_epochs
        self._stop_training = False

    def update(self, key_metric):
        raise NotImplementedError()

    def stop_training(self):
        return self._stop_training 


class EarlyStopping(AbstractStoppingCriterion):

    def __init__(self, epochs_to_wait,
                       larger_is_better,
                       running_mean_over_epochs=1,
                       **kwargs):
        super(EarlyStopping, self).__init__(**kwargs)
        self.epochs_to_wait = epochs_to_wait
        self.get_best = util.init_get_best(larger_is_better)
        self.epochs_done = 0
        self.epochs_since_last_improvement = 0
        self.running_mean_over_epochs = running_mean_over_epochs
        self.running_mean_vals = []

    def update(self, key_metric):
        if (len(self.running_mean_vals) == self.running_mean_over_epochs):
            self.running_mean_vals = self.running_mean_vals[1:]
        self.running_mean_vals.append(key_metric)
        key_metric = np.mean(self.running_mean_vals)
        self.epochs_done += 1
        
        new_best = self.get_best.process(self.epochs_done, key_metric)
        if (not new_best):
            self.epochs_since_last_improvement += 1 
        else:
            self.epochs_since_last_improvement = 0
        if (self.epochs_since_last_improvement == self.epochs_to_wait or
            self.epochs_done == self.max_epochs):
            self._stop_training = True
        

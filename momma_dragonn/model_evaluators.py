from sklearn.metrics import roc_auc_score
import numpy as np
import avutils.util as util
from collections import OrderedDict


class AbstractModelEvaluator(object):

    def get_key_metric_name(self):
        raise NotImplementedError() 

    def is_larger_better_for_key_metric(self):
        raise NotImplementedError() 

    def compute_key_metric(self, model_wrapper, data):
        raise NotImplementedError()

    def compute_all_stats(self, model_wrapper, data):
        raise NotImplementedError()


def remove_ambiguous_peaks(predictions, true_y): 
    indices_to_remove = np.where(true_y < 0)
    true_y_filtered = np.delete(true_y, indices_to_remove)
    predictions_filtered = np.delete(predictions, indices_to_remove)
    return predictions_filtered, true_y_filtered


def auroc_func(predictions, true_y):
    [num_rows, num_cols] = true_y.shape 
    aurocs=[]
    for c in range(num_cols): 
        true_for_task = true_y[:,c]
        predictions_for_task= predictions[:,c]
        predictions_for_task_filtered, true_y_for_task_filtered =\
         remove_ambiguous_peaks(predictions_for_task, true_y_for_task)
        task_auroc = roc_auc_score(true_y_for_task_filtered,
                                   predictions_for_task_filtered)
        aurocs.append(task_auroc) 
    return aurocs


AccuracyStats = util.enum(auROC="auROC")
compute_func_lookup = {
    AccuracyStats.auROC: auroc_func
}
is_larger_better_lookup = {
    AccuracyStats.auROC: True
}

class GraphAccuracyStats(AbstractModelEvaluator):

    def __init__(self, key_metric): #just auROC supported for now
        self.key_metric = key_metric 

    def get_key_metric_name(self):
        return self.key_metric

    def is_larger_better_for_key_metric(self):
        return is_larger_better_lookup[self.key_metric]

    def compute_key_metric(self, model_wrapper, data):
        predictions = model_wrapper.predict(train_data_for_eval.X)
        return self.compute_summary_stat(
                    per_output_stats=self.compute_per_output_stats(
                                      predictions, train_data_for_eval.Y,
                                      self.key_metric),
                    summary_op=np.mean) 

    def compute_summary_stat(self, per_output_stats, summary_op):
        to_summarise = [] 
        for stats in per_output_stats.values():
            to_summarise.extend(stats)
        return summary_op(to_summarise)

    def compute_per_output_stats(self, predictions, true_y, metric_name):
        func = compute_func_lookup[metric_name] 
        to_return = OrderedDict() 
        output_names = sorted(true_y.keys()) 
        for output_name in output_names:
            to_return[output_name] = func(predictions=predictions[output_name],
                                         true_y=true_y[output_name]) 
        return to_return

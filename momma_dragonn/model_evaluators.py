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
        true_y_for_task = true_y[:,c]
        predictions_for_task= predictions[:,c]
        predictions_for_task_filtered, true_y_for_task_filtered =\
         remove_ambiguous_peaks(predictions_for_task, true_y_for_task)
        task_auroc = roc_auc_score(y_true=true_y_for_task_filtered,
                                   y_score=predictions_for_task_filtered)
        aurocs.append(task_auroc) 
    return aurocs


def onehot_rows_crossent_func(predictions, true_y):
    #squeeze to get rid of channel axis 
    predictions, true_y = [np.squeeze(arr) for arr in [predictions, true_y]] 
    assert len(predictions.shape)==3
    assert len(true_y.shape)==3
    #transpose to put the row axis at the end
    predictions, true_y = [np.transpose(arr, (0,2,1)) for arr in
                            [predictions, true_y]] 
    #reshape
    predictions, true_y = [np.reshape(arr, (-1, 4)) for arr in
                            [predictions, true_y]]
    #clip
    predictions, true_y = [np.clip(arr, (10**-6), (1-(10**-6))) for arr in
                            [predictions, true_y]]
    #compute categ crossentropy
    return -np.sum(true_y*np.log(predictions))


AccuracyStats = util.enum(
    auROC="auROC",
    onehot_rows_crossent="onehot_rows_crossent")
compute_func_lookup = {
    AccuracyStats.auROC: auroc_func,
    AccuracyStats.onehot_rows_crossent:
        onehot_rows_crossent_func
}
is_larger_better_lookup = {
    AccuracyStats.auROC: True,
    AccuracyStats.onehot_rows_crossent: False
}

class GraphAccuracyStats(AbstractModelEvaluator):

    def __init__(self, key_metric, all_metrics): #just auROC supported for now
        self.key_metric = key_metric 
        self.all_metrics = all_metrics 

    def get_key_metric_name(self):
        return self.key_metric

    def is_larger_better_for_key_metric(self):
        return is_larger_better_lookup[self.key_metric]

    def compute_key_metric(self, model_wrapper, data, batch_size):
        predictions = model_wrapper.predict(data.X, batch_size)
        return self.compute_summary_stat(
                    per_output_stats=self.compute_per_output_stats(
                                      predictions=predictions,
                                      true_y=data.Y,
                                      metric_name=self.key_metric),
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


    def compute_all_stats(self, model_wrapper, data, batch_size):
        predictions = model_wrapper.predict(data.X, batch_size)
        all_stats = OrderedDict()
        for metric_name in self.all_metrics:
            per_output_stats = self.compute_per_output_stats( 
                                        predictions=predictions,
                                        true_y=data.Y,
                                        metric_name=metric_name)
            mean = self.compute_summary_stat(
                    per_output_stats=per_output_stats,
                    summary_op=np.mean)
            all_stats["per_output_"+metric_name] = per_output_stats
            all_stats["mean_"+metric_name] = mean
        return all_stats

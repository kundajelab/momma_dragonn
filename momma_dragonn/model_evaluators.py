from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
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


def binarycrossent_func(predictions, true_y):
    [num_rows, num_cols] = true_y.shape 
    binary_crossents=[]
    for c in range(num_cols): 
        true_y_for_task = true_y[:,c]
        predictions_for_task = predictions[:,c]
        predictions_for_task_filtered, true_y_for_task_filtered = \
         remove_ambiguous_peaks(predictions_for_task, true_y_for_task)

        #1E-7 is the "epsilon" used by keras as of April 23rd 2018
        #(see keras/backend/common.py)
        predictions_for_task_filtered =\
            np.maximum(0.0000001, predictions_for_task_filtered)
        predictions_for_task_filtered =\
            np.minimum(0.9999999, predictions_for_task_filtered)

        task_binarycrossent =\
            -np.mean(
                true_y_for_task_filtered
                *np.log(predictions_for_task_filtered)
              + (1-true_y_for_task_filtered)
                *np.log(1-predictions_for_task_filtered)) 

        binary_crossents.append(float(task_binarycrossent)) 
    return binary_crossents


def auroc_func(predictions, true_y):
    [num_rows, num_cols] = true_y.shape 
    aurocs=[]
    for c in range(num_cols): 
        true_y_for_task = true_y[:,c]
        predictions_for_task = predictions[:,c]
        predictions_for_task_filtered, true_y_for_task_filtered = \
         remove_ambiguous_peaks(predictions_for_task, true_y_for_task)
        #in case true_y_for_task_filtered is continuous "prob pos",
        # turn it into binary labels
        true_y_for_task_filtered = np.array([1 if x > 0.5 else 0 for x in
                                             true_y_for_task_filtered])
        task_auroc = roc_auc_score(y_true=true_y_for_task_filtered,
                                   y_score=predictions_for_task_filtered)
        aurocs.append(task_auroc) 
    return aurocs


def auprc_func(predictions, true_y):
    # sklearn only supports 2 classes (0,1) for the auPRC calculation 
    [num_rows, num_cols]=true_y.shape 
    auprcs=[]
    for c in range(num_cols): 
        true_y_for_task=np.squeeze(true_y[:,c])
        predictions_for_task=np.squeeze(predictions[:,c])
        predictions_for_task_filtered,true_y_for_task_filtered = \
         remove_ambiguous_peaks(predictions_for_task, true_y_for_task)
        true_y_for_task_filtered = np.array([1 if x > 0.5 else 0 for x in
                                             true_y_for_task_filtered])
        task_auprc = average_precision_score(true_y_for_task_filtered, predictions_for_task_filtered)
        auprcs.append(task_auprc) 
    return auprcs

def get_accuracy_stats_for_task(predictions, true_y, c):
    true_y_for_task=np.squeeze(true_y[:,c])
    predictions_for_task=np.squeeze(predictions[:,c])
    predictions_for_task_filtered,true_y_for_task_filtered = remove_ambiguous_peaks(predictions_for_task,true_y_for_task)
    predictions_for_task_filtered_round = np.array([round(el) for el in predictions_for_task_filtered])
    accuratePredictions = predictions_for_task_filtered_round==true_y_for_task_filtered

    numPositives_forTask=np.sum(true_y_for_task_filtered==1,axis=0,dtype="float")
    numNegatives_forTask=np.sum(true_y_for_task_filtered==0,axis=0,dtype="float")

    accuratePredictions_positives = np.sum(accuratePredictions*(true_y_for_task_filtered==1),axis=0)
    accuratePredictions_negatives = np.sum(accuratePredictions*(true_y_for_task_filtered==0),axis=0)

    returnDict = {
        'accuratePredictions': accuratePredictions,
        'numPositives_forTask': numPositives_forTask,
        'numNegatives_forTask': numNegatives_forTask,
        'true_y_for_task_filtered': true_y_for_task_filtered,
        'predictions_for_task_filtered': predictions_for_task_filtered,
        'accuratePredictions_positives': accuratePredictions_positives,
        'accuratePredictions_negatives': accuratePredictions_negatives
    }
    return returnDict


def unbalanced_accuracy(predictions, true_y):
    assert predictions.shape==true_y.shape
    assert len(predictions.shape)==2
    [num_rows, num_cols]=true_y.shape 
    unbalanced_accuracies = []
    for c in range(num_cols): 
        r = get_accuracy_stats_for_task(predictions, true_y, c)

        unbalancedAccuracy_forTask = (r['accuratePredictions_positives'] + r['accuratePredictions_negatives'])/(r['numPositives_forTask']+r['numNegatives_forTask']).astype("float")
        unbalanced_accuracies.append(unbalancedAccuracy_forTask) 
    return unbalanced_accuracies


def balanced_accuracy(predictions, true_y):
    assert predictions.shape==true_y.shape, ("Did you make sure your label "
           "data have the same dims as the model's output?")
    assert len(predictions.shape)==2
    [num_rows, num_cols]=true_y.shape 
    balanced_accuracies = [] 
    for c in range(num_cols): 
        r = get_accuracy_stats_for_task(predictions, true_y, c)
    
        positivesAccuracy_forTask = r['accuratePredictions_positives']/r['numPositives_forTask']
        negativesAccuracy_forTask = r['accuratePredictions_negatives']/r['numNegatives_forTask']

        balancedAccuracy_forTask= (positivesAccuracy_forTask+negativesAccuracy_forTask)/2
        balanced_accuracies.append(balancedAccuracy_forTask) 
    return balanced_accuracies


def spearman_corr(predictions, true_y):
    import scipy.stats
    #first return val is correlation, second return val is a p-value
    #get task-specific correlations 
    num_tasks=predictions.shape[1] 
    task_correlations_all=[] 
    task_pvalues_all=[] 
    for t in range(num_tasks): 
        task_cor,task_p=scipy.stats.spearmanr(predictions[:,t],true_y[:,t])
        task_correlations_all.append(task_cor) 
        task_pvalues_all.append(task_p) 
    return task_correlations_all

def pearson_corr(predictions, true_y):
    import scipy.stats
    #first return val is correlation, second return val is a p-value
    #get task-specific correlations 
    num_tasks=predictions.shape[1]
    task_correlations_all=[]
    task_pvalues_all=[]
    for t in range(num_tasks):
        task_cor,task_p=scipy.stats.pearsonr(predictions[:,t],true_y[:,t])
        task_correlations_all.append(np.asscalar(task_cor))
        task_pvalues_all.append(np.asscalar(task_p))
    return task_correlations_all

def mean_squared_error(predictions, true_y):
    from sklearn.metrics import mean_squared_error
    #first return val is correlation, second return val is a p-value
    #get task-specific correlations 
    num_tasks=predictions.shape[1]
    task_mses_all=[]
    for t in range(num_tasks):
        task_mse=mean_squared_error(true_y[:,t],predictions[:,t])
        task_mses_all.append(np.asscalar(task_mse))
    return task_mses_all


#for autoencoders
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
    predictions = np.clip(predictions, (10**-6), (1-(10**-6)))
    #compute categ crossentropy
    return [-np.mean(np.sum(true_y*np.log(predictions),axis=-1))]


AccuracyStats = util.enum(
    binary_crossent="binary_crossent",
    auROC="auROC",
    auPRC="auPRC",
    balanced_accuracy="balanced_accuracy",
    unbalanced_accuracy="unbalanced_accuracy",
    spearman_corr="spearman_corr",
    pearson_corr="pearson_corr",
    mean_squared_error="mean_squared_error",
    onehot_rows_crossent="onehot_rows_crossent")
compute_func_lookup = {
    AccuracyStats.binary_crossent: binarycrossent_func,
    AccuracyStats.auROC: auroc_func,
    AccuracyStats.auPRC: auprc_func,
    AccuracyStats.balanced_accuracy: balanced_accuracy,
    AccuracyStats.unbalanced_accuracy: unbalanced_accuracy,
    AccuracyStats.spearman_corr: spearman_corr,
    AccuracyStats.pearson_corr: pearson_corr,
    AccuracyStats.mean_squared_error: mean_squared_error,
    AccuracyStats.onehot_rows_crossent:
        onehot_rows_crossent_func
}
is_larger_better_lookup = {
    AccuracyStats.binary_crossent: False,
    AccuracyStats.auROC: True,
    AccuracyStats.auPRC: True,
    AccuracyStats.balanced_accuracy: True,
    AccuracyStats.unbalanced_accuracy: True,
    AccuracyStats.spearman_corr: True,
    AccuracyStats.pearson_corr: True,
    AccuracyStats.mean_squared_error: False,
    AccuracyStats.onehot_rows_crossent: False,
}


class GraphAccuracyStats(AbstractModelEvaluator):

    def __init__(self, key_metric, all_metrics):
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


class SequentialAccuracyStats(AbstractModelEvaluator):

    def __init__(self, key_metric, all_metrics, tasks_subset=None):
        self.key_metric = key_metric 
        if (":" in self.key_metric):
            self.core_key_metric_name = key_metric.split(":")[0]
        else:
            self.core_key_metric_name = key_metric
        self.all_metrics = all_metrics 
        self.tasks_subset = tasks_subset

    def get_key_metric_name(self):
        return self.key_metric

    def is_larger_better_for_key_metric(self):
        return is_larger_better_lookup[self.core_key_metric_name]

    def compute_key_metric(self, model_wrapper, data, batch_size):
        predictions = model_wrapper.predict(data.X, batch_size)
        tasks_subset = self.tasks_subset
        if (":" in self.key_metric):
            tasks_subset = [int(x) for x in
                            self.key_metric.split(":")[1].split(",")]
        if (tasks_subset is None):
            return np.mean(compute_func_lookup[self.core_key_metric_name](
                predictions=predictions,
                true_y=data.Y))
        else:
            return np.mean(compute_func_lookup[self.core_key_metric_name](
                            predictions=predictions[:,tasks_subset],
                            true_y=data.Y[:,tasks_subset]))

    def compute_all_stats(self, model_wrapper, data, batch_size):
        predictions = model_wrapper.predict(data.X, batch_size)
        all_stats = OrderedDict()
        for metric_name in self.all_metrics:
            tasks_subset = self.tasks_subset
            if (":" in metric_name):
                core_metric_name =  metric_name.split(":")[0]
                tasks_subset = [int(x) for x in
                                metric_name.split(":")[0].split(",")]
            else:
                core_metric_name = metric_name
            if (tasks_subset is None):
                per_output = compute_func_lookup[core_metric_name](
                                predictions=predictions[:,:],
                                true_y=data.Y[:,:])
            else:
                per_output = compute_func_lookup[core_metric_name](
                                predictions=predictions[:,tasks_subset],
                                true_y=data.Y[:,tasks_subset])
            mean = np.mean(per_output) 
            all_stats["per_output_"+metric_name] = per_output
            all_stats["mean_"+metric_name] = mean
        return all_stats

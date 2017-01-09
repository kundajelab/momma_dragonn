from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import avutils.util as util
from collections import OrderedDict, defaultdict
import sys 
import pdb 

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


def auroc_func(predictions, true_y,thresh=None):
    [num_rows, num_cols] = true_y.shape 
    aurocs=[]
    for c in range(num_cols): 
        true_y_for_task = true_y[:,c]
        predictions_for_task = predictions[:,c]
        predictions_for_task_filtered, true_y_for_task_filtered = \
         remove_ambiguous_peaks(predictions_for_task, true_y_for_task)
        try:
            task_auroc = roc_auc_score(y_true=true_y_for_task_filtered,
                                   y_score=predictions_for_task_filtered)
        except Exception as e:
            #if there is only one class in the batch of true_y, then auROC cannot be calculated
            print("Could not calculate auROC:")
            print(str(e))
            task_auroc=None 
        aurocs.append(task_auroc) 
    return aurocs

def auprc_func(predictions, true_y,thresh=None):
    # sklearn only supports 2 classes (0,1) for the auPRC calculation 
    [num_rows, num_cols]=true_y.shape 
    auprcs=[]
    for c in range(num_cols): 
        true_y_for_task=np.squeeze(true_y[:,c])
        predictions_for_task=np.squeeze(predictions[:,c])
        predictions_for_task_filtered,true_y_for_task_filtered = \
         remove_ambiguous_peaks(predictions_for_task, true_y_for_task)
        try:
            task_auprc = average_precision_score(true_y_for_task_filtered, predictions_for_task_filtered);
        except:
            print("Could not calculated auPRC:")
            print(sys.exc_info()[0])
            task_auprc=None 
        auprcs.append(task_auprc) 
    return auprcs;

def get_accuracy_stats_for_task(predictions, true_y, c):
    true_y_for_task=np.squeeze(true_y[:,c])
    predictions_for_task=np.squeeze(predictions[:,c])
    predictions_for_task_filtered,true_y_for_task_filtered = remove_ambiguous_peaks(predictions_for_task,true_y_for_task)
    predictions_for_task_filtered_round = np.array([round(el) for el in predictions_for_task_filtered])
    accuratePredictions = predictions_for_task_filtered_round==true_y_for_task_filtered;

    numPositives_forTask=np.sum(true_y_for_task_filtered==1,axis=0,dtype="float");
    numNegatives_forTask=np.sum(true_y_for_task_filtered==0,axis=0,dtype="float"); 

    accuratePredictions_positives = np.sum(accuratePredictions*(true_y_for_task_filtered==1),axis=0);
    accuratePredictions_negatives = np.sum(accuratePredictions*(true_y_for_task_filtered==0),axis=0);
    pdb.set_trace() 
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


def unbalanced_accuracy(predictions, true_y,thresh=None):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    unbalanced_accuracies = []
    for c in range(num_cols):
        r = get_accuracy_stats_for_task(predictions, true_y, c)
        unbalancedAccuracy_forTask = (r['accuratePredictions_positives'] + r['accuratePredictions_negatives'])/(r['numPositives_forTask']+r['numNegatives_forTask']).astype("float");
        unbalanced_accuracies.append(unbalancedAccuracy_forTask) 
    return unbalanced_accuracies;

def positives_accuracy(predictions,true_y,thresh=None):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    positive_accuracies = []
    for c in range(num_cols):
        r = get_accuracy_stats_for_task(predictions, true_y, c)
        positiveAccuracy_forTask = float(r['accuratePredictions_positives'])/float(r['numPositives_forTask'])
        positive_accuracies.append(positiveAccuracy_forTask) 
    return positive_accuracies;

def negatives_accuracy(predictions,true_y,thresh=None):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    negative_accuracies = []
    for c in range(num_cols):
        r = get_accuracy_stats_for_task(predictions, true_y, c)
        negativeAccuracy_forTask = float(r['accuratePredictions_negatives'])/float(r['numNegatives_forTask'])
        pdb.set_trace() 
        negative_accuracies.append(negativeAccuracy_forTask) 
    return negative_accuracies;

def num_positives(predictions,true_y,thresh=None):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    num_positives = []
    for c in range(num_cols):
        r = get_accuracy_stats_for_task(predictions, true_y, c)
        num_positives.append(r['numPositives_forTask']) 
    return num_positives;

def num_negatives(predictions,true_y,thresh=None):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    num_negatives = []
    for c in range(num_cols):
        r = get_accuracy_stats_for_task(predictions, true_y, c)
        num_negatives.append(r['numNegatives_forTask'])
    return num_negatives;

def balanced_accuracy(predictions, true_y, thresh=None):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    balanced_accuracies = []
    for c in range(num_cols): 
        r = get_accuracy_stats_for_task(predictions, true_y, c)
    
        positivesAccuracy_forTask = r['accuratePredictions_positives']/r['numPositives_forTask'];
        negativesAccuracy_forTask = r['accuratePredictions_negatives']/r['numNegatives_forTask'];

        balancedAccuracy_forTask= (positivesAccuracy_forTask+negativesAccuracy_forTask)/2;
        balanced_accuracies.append(balancedAccuracy_forTask) 
    return balanced_accuracies


def onehot_rows_crossent_func(predictions, true_y,thresh=None):
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

def prob_to_labels(predictions,thresh):
    class_labels=np.zeros_like(predictions)
    class_labels[class_labels >=thresh]=1 
    return class_labels 

def recall_at_fdr_function(predictions,true_y,thresh):
    if float(thresh)>1: 
        thresh=float(thresh)/100 
    [num_rows, num_cols]=true_y.shape
    recall_at_fdr_vals=[]
    for c in range(num_cols): 
        true_y_for_task=np.squeeze(true_y[:,c])
        predictions_for_task=np.squeeze(predictions[:,c])
        predictions_for_task_filtered,true_y_for_task_filtered = remove_ambiguous_peaks(predictions_for_task, true_y_for_task)

        #group by predicted prob
        predictedProbToLabels = defaultdict(list)
        for predictedProb, trueY in zip(predictions_for_task_filtered, true_y_for_task_filtered):
            predictedProbToLabels[predictedProb].append(trueY)
        #sort in ascending order of confidence
        sortedThresholds = sorted(predictedProbToLabels.keys())
        toReturnDict = OrderedDict();
        thresholdPairs=[("recallAtFDR"+str(thresh),thresh)]
        #sort desired recall thresholds by descending order of fdr        
        totalPositives = np.sum(true_y_for_task_filtered)
        totalNegatives = np.sum(1-true_y_for_task_filtered)
        #start at 100% recall
        confusionMatrixStatsSoFar = [[0,totalNegatives]
                                    ,[0,totalPositives]]
        recallsForThresholds = []; #for debugging
        fdrsForThresholds = [];

        for threshold in sortedThresholds:
            labelsAtThreshold=predictedProbToLabels[threshold];
            positivesAtThreshold=sum(labelsAtThreshold)
            negativesAtThreshold = len(labelsAtThreshold)-positivesAtThreshold
            
            #when you cross this threshold they all get predicted as negatives.
            confusionMatrixStatsSoFar[0][0] += negativesAtThreshold
            confusionMatrixStatsSoFar[0][1] -= negativesAtThreshold
            confusionMatrixStatsSoFar[1][0] += positivesAtThreshold
            confusionMatrixStatsSoFar[1][1] -= positivesAtThreshold
            totalPredictedPositives = confusionMatrixStatsSoFar[0][1]\
                                  + confusionMatrixStatsSoFar[1][1]
            fdr = 1 - (confusionMatrixStatsSoFar[1][1]/
                       float(totalPredictedPositives))\
                       if totalPredictedPositives > 0 else 0.0
            recall = confusionMatrixStatsSoFar[1][1]/float(totalPositives)
            recallsForThresholds.append(recall)
            fdrsForThresholds.append(fdr)
            #first index of a thresholdPair is the name, second idx
            #is the actual threshold
            while (len(thresholdPairs)>0 and fdr<=thresholdPairs[0][1]):
                toReturnDict[thresholdPairs[0][0]]=recall
                thresholdPairs=thresholdPairs[1::]
            if len(thresholdPairs)==0:
                break;
        for thresholdPair in thresholdPairs:
            toReturnDict[thresholdPairs[0][0]]=0.0
        return [toReturnDict['recallAtFDR'+str(thresh)]]

AccuracyStats = util.enum(
    auROC="auROC",
    auPRC="auPRC",
    balanced_accuracy="balanced_accuracy",
    unbalanced_accuracy="unbalanced_accuracy",
    positives_accuracy="positives_accuracy",
    negatives_accuracy="negatives_accuracy",
    num_positives="num_positives",
    num_negatives="num_negatives",
    onehot_rows_crossent="onehot_rows_crossent",
    recall_at_fdr="recallAtFDR")

compute_func_lookup = {
    AccuracyStats.auROC: auroc_func,
    AccuracyStats.auPRC: auprc_func,
    AccuracyStats.balanced_accuracy: balanced_accuracy,
    AccuracyStats.unbalanced_accuracy: unbalanced_accuracy,
    AccuracyStats.onehot_rows_crossent:onehot_rows_crossent_func,
    AccuracyStats.recall_at_fdr: recall_at_fdr_function,
    AccuracyStats.positives_accuracy:positives_accuracy,
    AccuracyStats.negatives_accuracy:negatives_accuracy,
    AccuracyStats.num_positives:num_positives,
    AccuracyStats.num_negatives:num_negatives
}
is_larger_better_lookup = {
    AccuracyStats.auROC: True,
    AccuracyStats.auPRC: True,
    AccuracyStats.balanced_accuracy: True,
    AccuracyStats.unbalanced_accuracy: True,
    AccuracyStats.onehot_rows_crossent: False,
    AccuracyStats.recall_at_fdr: True
}

multi_level_metrics=["recallAtFDR"]

class GraphAccuracyStats(AbstractModelEvaluator):

    def __init__(self, key_metric, all_metrics): #just auROC supported for now
        self.key_metric = key_metric 
        self.all_metrics = all_metrics 

    def get_key_metric_name(self):
        return self.key_metric

    def is_larger_better_for_key_metric(self):
        return is_larger_better_lookup[self.key_metric]

    #uses a generators to do batch predictions
    def get_model_predictions(self,model_wrapper,data_generator,samples_to_use,batch_size_predict):
        print("getting model predictions!!")
        data_batch=next(data_generator)
        x=data_batch[0]
        y=data_batch[1]
        new_batch_predictions=model_wrapper.predict(x,50)
        pdb.set_trace() 
        #print("y.shape:"+str(y.shape))
        #print("new_batch_predictions.shape:"+str(new_batch_predictions.shape))
        true_y={}
        predictions={} 
        samples_used=0
        #pre-allocate numpy arrays to avoid running slow numpy concatenation operations
        sample_output_name=y.keys()[0]
        y_shape=y[sample_output_name].shape 
        for output_name in y.keys():
            true_y[output_name]=np.zeros((samples_to_use,y_shape[1]))
            true_y[output_name][samples_used:samples_used+y_shape[0]]=y[output_name]
            predictions[output_name]=np.zeros((samples_to_use,y_shape[1]))
            predictions[output_name][samples_used:samples_used+y_shape[0]]=new_batch_predictions[output_name] 
        samples_used+=y_shape[0]
        print(str(samples_used))
        while len(x.values())>0:
            data_batch=next(data_generator)
            x=data_batch[0]
            y=data_batch[1]
            if len(x.values())==0:
                break
            new_batch_predictions=model_wrapper.predict(x,50)
            y_shape=y[sample_output_name].shape 
            for output_name in y.keys():
                true_y[output_name][samples_used:samples_used+y_shape[0]]=y[output_name]
                predictions[output_name][samples_used:samples_used+y_shape[0]]=new_batch_predictions[output_name]
            samples_used+=y_shape[0]
            print(str(samples_used))
        return predictions,true_y
    
    def compute_key_metric(self, model_wrapper, data_generator, samples_to_use, batch_size):
        print("calling compute_key_metric:") 
        predictions,true_y=self.get_model_predictions(model_wrapper,data_generator,samples_to_use,batch_size)
        print("got predictions!!") 
        return self.compute_summary_stat(
                    per_output_stats=self.compute_per_output_stats(
                                      predictions=predictions,
                                      true_y=true_y,
                                      metric_name=self.key_metric),
                    summary_op=np.mean) 

    def compute_summary_stat(self, per_output_stats, summary_op):
        print("computing summary stat:") 
        to_summarise = []
        for stats in per_output_stats.values():
            to_summarise.extend(stats)
        while None in to_summarise:
            to_summarise.remove(None) 
        return summary_op(to_summarise)

    def split_metric_name(self,metric_name):
        stripped_metric_name=metric_name
        thresh=None
    
        for multi_level_metric_name in multi_level_metrics: 
            if metric_name.startswith(multi_level_metric_name):
                stripped_metric_name=multi_level_metric_name
                thresh=int(metric_name.replace(multi_level_metric_name,""))
        return stripped_metric_name,thresh 
    
    def compute_per_output_stats(self, predictions, true_y, metric_name):
        metric_name,metric_thresh=self.split_metric_name(metric_name) 
        func = compute_func_lookup[metric_name]
        to_return = OrderedDict() 
        output_names = sorted(true_y.keys()) 
        for output_name in output_names:
            print("computing stat for output:"+str(output_name)) 
            to_return[output_name] = func(predictions=predictions[output_name],true_y=true_y[output_name],thresh=metric_thresh) 
        return to_return


    def compute_all_stats(self, model_wrapper, data_generator, samples_to_use, batch_size):
        predictions,true_y=self.get_model_predictions(model_wrapper,data_generator,samples_to_use,batch_size)
        all_stats = OrderedDict()
        for metric_name in self.all_metrics:
            per_output_stats = self.compute_per_output_stats( 
                                        predictions=predictions,
                                        true_y=true_y,
                                        metric_name=metric_name)
            mean = self.compute_summary_stat(
                    per_output_stats=per_output_stats,
                    summary_op=np.mean)
            all_stats["per_output_"+metric_name] = per_output_stats
            all_stats["mean_"+metric_name] = mean
        return all_stats

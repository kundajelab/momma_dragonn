import argparse
import yaml 
import h5py 
import keras
from keras.legacy.models import *
from momma_dragonn.model_evaluators import *
import pickle
import numpy as np 
import pdb

def get_predictions(hdf5_source,batch_size,individual_task_output_shape,output_task_names,model):
    num_generated=0
    total_entries=hdf5_source.values()[0].shape[0]
    input_modes=hdf5_source.keys() 
    print("total entries:"+str(total_entries))
    predictions={}
    for task in output_task_names:
        predictions[task]=np.zeros(individual_task_output_shape)
    print("initialized output dictionary for predictions")    
    while num_generated < total_entries:
        print str(num_generated) 
        start_index=num_generated
        end_index=min([total_entries,start_index+batch_size])
        x_batch={}  
        for input_mode in input_modes: 
            x_batch[input_mode] = hdf5_source[input_mode][start_index:end_index]
        predictions_batch=model.predict(x_batch)
        #add the predictions to the dictionary
        for task in output_task_names:
            predictions[task][start_index:end_index]=predictions_batch[task]
        num_generated+=(end_index-start_index)
    return predictions 


def parse_args():
    parser=argparse.ArgumentParser(description='Provide a model yaml & weights files & a dataset, get model predictions and accuracy metrics')
    parser.add_argument('--yaml',help='yaml file that stores model architecture')
    parser.add_argument('--weights',help='hdf5 file that stores model weights')
    parser.add_argument('--data',help='hdf5 file that stores the data')
    parser.add_argument('--predictions_pickle',help='name of pickle to save predictions')
    parser.add_argument('--accuracy_metrics_file',help='file name to save accuracy metrics')
    parser.add_argument('--predictions_pickle_to_load',help="if predictions have already been generated, provide a pickle with them to just compute the accuracy metrics",default=None)
    parser.add_argument('--batch_size',type=int,help='batch size to use to make model predictions',default=50)
    return parser.parse_args()

def main():
    args=parse_args()
    #get the data
    data=h5py.File(args.data,'r')
    inputs=data['X']
    outputs=data['Y']

    if args.predictions_pickle_to_load==None: 
        #get the model 
        yaml_string=open(args.yaml,'r').read()
        model_config=yaml.load(yaml_string)
        model=Graph.from_config(model_config)
        print("got model architecture")

        #load the model weights
        model.load_weights(args.weights)
        print("loaded model weights")

        #get the model predictions in a batch-like manner
        batch_size=args.batch_size

        output_task_names=outputs.keys()
        individual_task_output_shape=outputs.values()[0].shape
        predictions=get_predictions(inputs,args.batch_size,individual_task_output_shape,output_task_names,model)
        #predictions=model.predict(inputs)
        print('got model predictions')

        #pickle the predictions in case an error occurs downstream
        #this will allow for easy recovery of model predictions without having to regenerate them
        with open(args.predictions_pickle,'wb') as handle:
            pickle.dump(predictions,handle,protocol=pickle.HIGHEST_PROTOCOL)
        print("pickled the model predictions to file:"+str(args.predictions_pickle))
    else:
        with open(args.predictions_pickle_to_load,'rb') as handle:
             predictions=pickle.load(handle)

    print('computing accuracy metrics...')
    recallAtFDR50=dict()
    recallAtFDR20=dict()
    auroc_vals=dict()
    auprc_vals=dict()
    unbalanced_accuracy_vals=dict()
    balanced_accuracy_vals=dict()
    positives_accuracy_vals=dict()
    negatives_accuracy_vals=dict()
    num_positive_vals=dict()
    num_negative_vals=dict()
    
    for output_mode in predictions: 
        #compute the accuracy metrics
        recallAtFDR50[output_mode]=recall_at_fdr_function(predictions[output_mode],outputs[output_mode],50)
        print('got recall at FDR50!') 
        recallAtFDR20[output_mode]=recall_at_fdr_function(predictions[output_mode],outputs[output_mode],20)
        print('got recall at FDR20!')
        auroc_vals[output_mode]=auroc_func(predictions[output_mode],outputs[output_mode])
        print('got auROC vals!')
        auprc_vals[output_mode]=auprc_func(predictions[output_mode],outputs[output_mode])
        print('got auPRC vals!')
        unbalanced_accuracy_vals[output_mode]=unbalanced_accuracy(predictions[output_mode],outputs[output_mode])
        print('got unbalanced accuracy')
        balanced_accuracy_vals[output_mode]=balanced_accuracy(predictions[output_mode],outputs[output_mode])
        print('got balanced accuracy')
        positives_accuracy_vals[output_mode]=positives_accuracy(predictions[output_mode],outputs[output_mode])
        print('got positives accuracy')
        negatives_accuracy_vals[output_mode]=negatives_accuracy(predictions[output_mode],outputs[output_mode])
        print('got negative accuracy vals')
        num_positive_vals[output_mode]=num_positives(predictions[output_mode],outputs[output_mode])
        print('got number of positive values')
        num_negative_vals[output_mode]=num_negatives(predictions[output_mode],outputs[output_mode])
        print('got number of negative values')

    #write accuracy metrics to output file: 
    print('writing accuracy metrics to file...')
    outf=open(args.accuracy_metrics_file,'w')
    for key in recallAtFDR50.keys():
        outf.write('recallAtFDR50\t'+str(key)+'\t'+'\t'.join([str(i) for i in recallAtFDR50[key]])+'\n')
    for key in recallAtFDR20.keys():
        outf.write('recallAtFDR20\t'+str(key)+'\t'+'\t'.join([str(i) for i in recallAtFDR20[key]])+'\n')
    for key in auroc_vals.keys():
        outf.write('auroc_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in auroc_vals[key]])+'\n')
    for key in auprc_vals.keys():
        outf.write('auprc_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in auprc_vals[key]])+'\n')
    for key in unbalanced_accuracy_vals.keys():
        outf.write('unbalanced_accuracy_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in unbalanced_accuracy_vals[key]])+'\n')
    for key in balanced_accuracy_vals.keys():
        outf.write('balanced_accuracy_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in balanced_accuracy_vals[key]])+'\n')
    for key in positives_accuracy_vals.keys():
        outf.write('positives_accuracy_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in positives_accuracy_vals[key]])+'\n')
    for key in negatives_accuracy_vals.keys():
        outf.write('negatives_accuracy_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in negatives_accuracy_vals[key]])+'\n')    
    for key in num_positive_vals.keys():
        outf.write('num_positive_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in num_positive_vals[key]])+'\n')
    for key in num_negative_vals.keys():
        outf.write('num_negative_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in num_negative_vals[key]])+'\n')
    

if __name__=="__main__":
    main()
    

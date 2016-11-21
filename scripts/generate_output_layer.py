#THIS IS A HACK! TO BE REPLACED LATER
#generates the output layer (dense layer + sigmoid + output) for models with large numbers of output tasks
import json
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="Generate augmented json file with dense layers + activation functions for output tasks")
    parser.add_argument("--hyperparameter_input",help="hyperparameter_configs_list.yaml base file")
    parser.add_argument("--labels_input",help="labels.yaml file listing all output tasks")
    parser.add_argument("--hyperparameter_output",help="hyperparamter agumented file to be generated")
    return parser.parse_args() 

def get_label_list(labels_input):
    label_list=[]
    for entry in labels_input:
        label_list.append(entry.values()[0]["output_mode_name"])
    return label_list 

def main():
    args=parse_args()
    data=open(args.hyperparameter_input,'r').read()
    hp_input=json.loads(data)
    data=open(args.labels_input,'r').read()
    labels_input=json.loads(data)
    #get the list of output tasks 
    label_list=get_label_list(labels_input) 
    numtasks=len(label_list)
    print str(numtasks)
    #generate architecture output layers for each of the tasks.
    #replicate the final layer & pre-activation layer
    kwargs=hp_input[0]['model_creator']['kwargs']['nodes_config']
    pre_act=kwargs[-1]
    last_layer=kwargs[-2]
    kwargs_truncated=kwargs[0:-2]
    outputs_config=[]
    cur_name=last_layer['name']
    pre_act_name=pre_act['name'] 
    for task in range(numtasks):
        #copy the final activation layer
        last_layer_cur_task=last_layer
        last_layer_cur_task['name']=cur_name+'_'+str(task)
        last_layer_cur_task['kwargs']['output_dim']=1
        #add a pre-output activation layer
        pre_act_cur_task=pre_act
        pre_act_cur_task['input_name']=cur_name+'_'+str(task) 
        pre_act_cur_task['name']=pre_act_name+'_'+str(task)
        #update the list of output layers
        kwargs_truncated.append(last_layer_cur_task)
        kwargs_truncated.append(pre_act_cur_task) 
        outputs_config.append({'name':str(label_list[task]),'input_name':pre_act_cur_task['name']})
    hp_input[0]['model_creator']['kwargs']['nodes_config']=kwargs_truncated
    hp_input[0]['model_creator']['kwargs']['outputs_config']=outputs_config 
    #write the updated json output file
    output_yaml=open(args.hyperparameter_output,'w')
    json.dump(hp_input,output_yaml,indent=4,separators=(',',':'))

if __name__=="__main__":
    main() 

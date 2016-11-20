#splits a matrix of labels or weights into a 'one output file per task' format
#that can be used with Graph Models
#generates a labels.yaml & weights.yaml file
import argparse
import json
import numpy as np
from os.path import *


def parse_args():
    parser=argparse.ArgumentParser(description='Read in a matrix of labels (or weights) and split into individual output task files & corresponding yaml')
    parser.add_argument('--inputf',help='input matrix, first row contains task names, should be tab-separated')
    parser.add_argument('--output_dir',help='directory where output files should be stored')
    parser.add_argument('--output_yaml',help='full path to output yaml file')
    parser.add_argument('--category',help='one of "labels" or "weights"')
    return parser.parse_args()

def main():
    args=parse_args()
    data=open(args.inputf,'r').read().strip().split('\n')
    header=data[0].split('\t')
    num_tasks=len(header)-1 #first is the peak id
    task_dict=dict()
    yaml_contents=[]
    if args.output_dir.endswith('/')==False:
        args.output_dir=args.output_dir+'/'
    for i in range(1,len(header)):
        task_dict[header[i]]=open(args.output_dir+args.category+'.'+header[i],'w')
        task_dict[header[i]].write('Peak\t'+str(header[i])+'\n')
        yaml_dict=dict() 
        yaml_dict[args.category]=dict()
        yaml_dict[args.category]['output_mode_name']=header[i]
        yaml_dict[args.category]['file_name']=args.output_dir+args.category+'.'+header[i]
        yaml_contents.append(yaml_dict)
        
    for line in data[1::]:
        tokens=line.split('\t')
        peak_name=tokens[0]
        for i in range(1,len(tokens)):
            task_name=header[i]
            val=tokens[i]
            task_dict[header[i]].write(peak_name+'\t'+val+'\n')
    
    output_yaml=open(args.output_yaml,'w')
    json.dump(yaml_contents,output_yaml,indent=4,separators=(',',':'))

if __name__=="__main__":
    main() 



#GENERATES A WEIGHTS FILE FOR SEQUENTIAL DATA INPUTS WITH MOMMA DRAGONN
#THIS REPLACES THE W0 & W1 PER TASK VALUES IN THE EARLIER FRAMEWORK
import argparse 
def parse_args():
    parser = argparse.ArgumentParser(description='use w1 and w0 per-task values to generate a weights matrix for momma dragonn')
    parser.add_argument('--labels_file',help="labels file") 
    parser.add_argument('--w1_file',help="w1 labels file with entries of the form 'task\tw1'")
    parser.add_argument('--w0_file',help="w0 labels file with entries of the form 'task\tw0'")
    parser.add_argument('--output_file',help="output filename")
    return parser.parse_args() 

def main():
    args=parse_args()
    data=open(args.labels_file,'r').read().strip().split('\n')
    tasks=data[0].split('\t')
    print(str(tasks))
    outf=open(args.output_file,'w')
    
    #generate dictionary of w1 weights
    w1_weights=open(args.w1_file,'r').read().strip().split('\n')
    w1_dict=dict()
    for line in w1_weights:
        tokens=line.split('\t')
        task=tokens[0]
        weights=round(float(tokens[1]),3) 
        w1_dict[task]=weights
        
    #generate dictionary of w0 weights
    w0_weights=open(args.w0_file,'r').read().strip().split('\n')
    w0_dict=dict()
    for line in w0_weights:
        tokens=line.split('\t')
        task=tokens[0]
        weights=round(float(tokens[1]),3)
        w0_dict[task]=weights

    outf.write(data[0]+'\n')
    counter=0 
    for line in data[1::]:
        counter+=1
        if counter % 10000==0:
            print str(counter)
        tokens=line.split('\t')
        peak_name=tokens[0]
        vals=[peak_name]
        for i in range(1,len(tokens)):
            task_name=tasks[i]
            multiplier1=w1_dict[task_name]
            multiplier0=w0_dict[task_name]
            curval=int(tokens[i])
            if curval==1:
                vals.append(multiplier1)
            elif curval==0:
                vals.append(multiplier0)
            else:
                vals.append(curval) 
        outf.write('\t'.join([str(i) for i in vals])+'\n')
        

    
    

if __name__=="__main__":
    main() 



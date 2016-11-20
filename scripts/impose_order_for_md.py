import argparse
#THIS SCRIPT ORDERS THE FASTA, TRAINING, TEST, VALID SPLIT FILES IN ACCORDANCE WITH THE FASTA FILES ENTRIES
def parse_args():
    parser = argparse.ArgumentParser(description='Impose order on deep learning FASTA and labels file')
    parser.add_argument('--fasta_file',help="fasta file")
    parser.add_argument('--labels_file',help="labels file") 
    parser.add_argument('--train_split',help="training labels")
    parser.add_argument('--valid_split',help="validation labels")
    parser.add_argument('--test_split',help="test labels")
    return parser.parse_args() 

def main():
    args=parse_args()
    #READ IN THE LABELS
    label_data=open(args.labels_file,'r').read().strip().split('\n')
    label_header=label_data[0] 
    #CREATE DICTIONARY OF LABELS
    label_dict=dict()
    for line in label_data[1::]:
        tokens=line.split('\t')
        label=tokens[0]
        label_dict[label]=line
    #print str(label_dict.keys())
    #READ IN THE TRAIN SPLIT
    train_dict=dict()
    train_data=open(args.train_split,'r').read().strip().split('\n')
    for line in train_data:
        train_dict[line]=1 
    #READ IN THE VALIDATION SPLIT
    valid_dict=dict()
    valid_data=open(args.valid_split,'r').read().strip().split('\n')
    for line in valid_data:
        valid_dict[line]=1
    #READ IN THE TEST SPLIT 
    test_dict=dict()
    test_data=open(args.test_split,'r').read().strip().split('\n') 
    for line in test_data:
        test_dict[line]=1

    #DO THE SORT AS FASTA IS READ IN!
    outf_labels=open(args.labels_file+'.sorted','w')
    outf_labels.write(label_header+'\n') 
    outf_train=open(args.train_split+'.sorted','w')
    outf_valid=open(args.valid_split+'.sorted','w')
    outf_test=open(args.test_split+".sorted",'w') 
    fasta_file=open(args.fasta_file,'r')
    counter=0 
    for line in fasta_file:
        if line.startswith('>'):
            peak_name=line[1:-1]
            counter+=1
            if counter%100000==0:
                print str(counter)
            try:
                outf_labels.write(label_dict[peak_name]+'\n')
            except:
                print("peak not found in label_dict:"+str(peak_name))
                continue
            if peak_name in train_dict:
                outf_train.write(peak_name+'\n')
            elif peak_name in test_dict:
                outf_test.write(peak_name+'\n')
            elif peak_name in valid_dict:
                outf_valid.write(peak_name+'\n')
                
if __name__=="__main__":
    main() 

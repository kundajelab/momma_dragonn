#reference sequence 
python make_dl_inputs.py --variant_bed pos_of_interest.csv --seq_size 2000 --out_prefix ref.seq.inputs

#introducing variants 
python make_dl_inputs.py --variant_bed pos_of_interest.csv --seq_size 2000 --out_prefix var.seq.inputs --incorporate_var
#!/usr/bin/env bash

##simdata generation:
./densityMotifSimulation.py --prefix gata --motifNames GATA_disc1 --max-motifs 3 --min-motifs 1 --mean-motifs 2 --seqLength 200 --numSeqs 1000
./densityMotifSimulation.py --prefix tal --motifNames TAL1_known1 --max-motifs 3 --min-motifs 1 --mean-motifs 2 --seqLength 200 --numSeqs 1000
./densityMotifSimulation.py --prefix talgata --motifNames GATA_disc1 TAL1_known1 --max-motifs 3 --min-motifs 1 --mean-motifs 2 --seqLength 200 --numSeqs 1000

#transfer files to appropriate dir

###
echo $'id\tgata\ttal' > labels.txt
zcat DensityEmbedding_prefix-gata_motifs-GATA_disc1_min-1_max-3_mean-2_zeroProb-0_seqLength-200_numSeqs-1000.simdata.gz | perl -lane 'if ($. > 1) {print "$F[0]\t1\t0"}' >> labels.txt
zcat DensityEmbedding_prefix-tal_motifs-TAL1_known1_min-1_max-3_mean-2_zeroProb-0_seqLength-200_numSeqs-1000.simdata.gz | perl -lane 'if ($. > 1) {print "$F[0]\t0\t1"}' >> labels.txt
zcat DensityEmbedding_prefix-talgata_motifs-GATA_disc1+TAL1_known1_min-1_max-3_mean-2_zeroProb-0_seqLength-200_numSeqs-1000.simdata.gz | perl -lane 'if ($. > 1) {print "$F[0]\t1\t1"}' >> labels.txt
gzip labels.txt

zcat labels.txt.gz | perl -lane 'if ($.%10 !=1 and $.%10 != 2) {print $F[0]}' | gzip -c > splits/train.txt.gz
zcat labels.txt.gz | perl -lane 'if ($.%10==1 and $. > 1) {print $F[0]}' | gzip -c > splits/valid.txt.gz
zcat labels.txt.gz | perl -lane 'if ($.%10==2) {print $F[0]}' | gzip -c > splits/test.txt.gz


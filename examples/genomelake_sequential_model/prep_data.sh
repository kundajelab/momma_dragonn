#!/usr/bin/env bash

zcat ~/genomelake/examples/JUND.HepG2.chr22.101bp_intervals.tsv.gz | perl -lane 'BEGIN {$saw_pos = 0} {if ($F[3]==1 || $F[3]==-1) {print $_; $saw_pos = 1} else {if ($F[3]==0 && $saw_pos) {print $_; $saw_pos = 0}}}' | gzip -c > subset_JUND.HepG2.chr22.101bp_intervals.tsv.gz
zcat subset_JUND.HepG2.chr22.101bp_intervals.tsv.gz | head -2000 | gzip -c > train_JUND.HepG2.chr22.101bp_intervals.tsv.gz
zcat subset_JUND.HepG2.chr22.101bp_intervals.tsv.gz | tail -916 | gzip -c > valid_JUND.HepG2.chr22.101bp_intervals.tsv.gz
rm subset_JUND.HepG2.chr22.101bp_intervals.tsv.gz

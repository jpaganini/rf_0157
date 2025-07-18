#!/bin/bash

conda activate phastaf
phastaf --outdir ../../results/18_phastaf/ E116508_complete_genome.fasta 
phastaf --outdir ../../results/18_phastaf/E116508 E116508_complete_genome.fasta 

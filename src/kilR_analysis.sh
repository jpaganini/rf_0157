#!/bin/bash

conda activate mafft
#MAFFT commands 
#kilR AA 
mafft --preservecase 2025_03_all_kilR_aa.fasta > 2025_03_MSA_AA_ALL_output_mafft.fasta

#kilR upstream
mafft --adjustdirection --preservecase kilR_upstream_regions_headers.fasta > 2025_03_MSA_tree_kilR_upstream.fasta
mafft --adjustdirection --preservecase 2025_03_clusters_features.fasta > 2025_03_MSA_clusters_features.fasta

kilR NT
mafft --adjustdirection --preservecase 2025_03_MSA_input.fasta > 2025_03_MSA_output.fasta 


conda activate iqtree
#iqtree commands
iqtree -s 2025_03_MSA_AA_ALL_output_mafft_modified.fasta
iqtree -s ../2025_03_MSA_tree_kilR_upstream.fasta

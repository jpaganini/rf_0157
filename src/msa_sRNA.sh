#!/bin/bash
#Get the sequences from Sakai and EDL933, which don't need reversing
samtools faidx ../data/04_reference_genomes/Sakai_GCF_000008865.2_ASM886v2_genomic.fna NC_002695.2:1268347-1268681 > ../results/17_msa/input_supp_figure/Sakai_sRNA.fasta
samtools faidx ../data/04_reference_genomes/EDL933_complete.fasta AE005174.2:1353530-1353864 > ../results/17_msa/input_supp_figure/EDL933_sRNA.fasta

#Get the sequences from the reference England phages, which will need to be reversed
samtools faidx -i ../data/05_reference_phages/315176_ONT.fasta 315176_ONT:38080-38414 > ../results/17_msa/input_supp_figure/315176_ONT_sRNA_reversed.fasta
samtools faidx -i ../data/05_reference_phages/E30228_ONT.fasta E30228_ONT:38186-38520 > ../results/17_msa/input_supp_figure/E30228_ONT_sRNA_reversed.fasta
samtools faidx -i ../data/05_reference_phages/267849_ONT.fasta 267849_ONT:30540-30874 > ../results/17_msa/input_supp_figure/267849_ONT_sRNA_reversed.fasta
samtools faidx -i ../data/05_reference_phages/E116508_stx2c_phage.fasta E116508_stx2c_sbcB:32706-33040 > ../results/17_msa/input_supp_figure/E116508_sRNA_reversed.fasta

#cat all files together
cat ../results/17_msa/input_supp_figure/*.fasta > ../results/17_msa/2025_02_all_sRNA_regions.fasta
cat ../results/17_msa/input_sequences/Feature* >> ../results/17_msa/2025_02_all_sRNA_regions.fasta

#Manually modified the headers

#Run Alignment
clustalo -i ../results/17_msa/2025_02_all_sRNA_regions_MODIFIED_HEADERS.fasta -o ../results/17_msa/2025_02_MSA_all_sRNA_regions.fasta --threads 8

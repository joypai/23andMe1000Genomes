# 23and1000Genomes
computational genomics project

## files

###svm.py: 

	
**Input**: VCF file 

**Output**: SVM predictions for each segment for each individual

**Usage**: `python svm.py chr.vcf labels ind_labelled -t test | train admix [admixed_inds]`

 - `chr.vcf` = VCF file

 - `labels` = file with list of population labels

 - `ind_labelled` = file with list of individual IDs with corresponding population labels (used for training)

 - specify type `-t`: `test`, `train`, or `admix`

 - if `admix`, please include path to a file with list of sample IDs for individuals in the given population `[admixed_inds]`

####Example for testing:
`python svm.py chr22_1000SNPs_filtered_3window.txt labels_specific.txt ind_labels_specific.txt -t test`

####Example for admixed:
`python svm.py chr22_1000SNPs_filtered_3window.txt labels_continental.txt ind_labels_continental.txt -t admix ACB_inds.txt`


----------
###hmm.py:

**Input**: SVM predictions for each segment

**Output**: "smoothed" predictions with assignment probabilities

**Usage**: `python svm.py predictions.txt labels.txt`

 - `predictions.txt` = population label predictions for each segment

 - `labels` = file with list of population labels

####Example:
`python hmm.py predictions6.txt labels_continental.txt`


---------
###ir.py: 

Given probabilities of the HMM's predictions for each segment for a given individual, fit isotonic regression model to recalibrate probabilities

**Usage**: `python ir.py prob_file`

 - `prob_file` = file with HMM probabilities for each segment label, separated by tabs

####Example: 
`python ir.py ind1_probabilities.txt`

##sample files

- `chr22_1000SNPs_filtered_1window.txt`	
    - truncated phased VCF file for chromosome 22 with rare variants excluded
	 - full data for all chromosomes can be found at [here](http://bochet.gcc.biostat.washington.edu/beagle/1000_Genomes_phase3_v5a/individual_chromosomes/)

##system requirements

 - Python packages:
     - **sklearn**
     - **matplotlib**

###
# svm.py
# COMS 4761 Project: 23and1000Genomes 
# Katie Lin, Joy Pai
###

import sys
from sklearn import svm
import numpy as np
from scipy.special import comb
from bitarray import bitarray
from sklearn import cross_validation
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
# import matplotlib.pyplot as plt

def process_vcf(chr_vcf):
	"""
	input: phased vcf file
	output: 
		1. individuals = list of individuals
		2. F = matrix indexed by individual IDs (HG...)
			F[HG] = list of SNP windows
			each window is a tuple of (hap1, hap2) 

		i.e. for individual HG00096
			F[HG00096] = [('00000','00001'), ('00100','00000'), ... ]
	"""
	individuals = []	 	# list of individuals (HG...)
	F = {}				# matrix holding SNP window info indexed by individual IDs
	M = 50				# window size = number of SNPS

	with open(chr_vcf,"r") as infile:
		line_count = 1
		SNP_window_count = -1
		SNP_line_count = 0

		for line in infile:
			# skip 4 un-needed header lines
			if line_count <= 4:
				line_count = line_count + 1
				continue

			# header line 5 with info about individuals
			elif line_count == 5:
				fields = line.strip().split("\t")
				individuals = fields[9:]

				# initialize F matrix
				F = initialize(individuals)
				
				line_count = line_count + 1
				continue

			# in actual SNP lines	
			else:
				fields = line.split("\t")
				SNP_pos = fields[1]
				SNP_id = fields[2]
				cur_snp_GT = fields[9:]		# GT info 

				# CHECK: num_individuals consistent with individuals list
				num_individuals = len(cur_snp_GT)
				if (num_individuals != len(individuals)):
					print "error: number of individuals not consistent"
					sys.exit(1)

				# seen a full SNP window start new window
				if SNP_line_count % M == 0:
					SNP_window_count = SNP_window_count + 1
					SNP_line_count = SNP_line_count + 1
					
					#for i in (0,1,2):
					for i in range(0,num_individuals):
						cur_ind = individuals[i] 		# cur_ind = HG format id
						phased = cur_snp_GT[i].split('|')
						hap1 = phased[0].strip()
						hap2 = phased[1].strip()

						# update F matrix
						new_window = (hap1, hap2)
						F[cur_ind].append(new_window)
						
				# still within current SNP window
				elif SNP_line_count % M != 0:
					SNP_line_count = SNP_line_count + 1
					
					#for i in (0,1,2):
					for i in range(0,num_individuals):
						cur_ind = individuals[i]		# cur_ind = HG format id
						phased = cur_snp_GT[i].split('|')
						cur_hap1 = phased[0].strip()	# 0 or 1
						cur_hap2 = phased[1].strip()	# 0 or 1

						# get hap1 and hap2 string for current SNP window
						(old_hap1, old_hap2) = F[cur_ind][SNP_window_count]

						# append new hap1,hap2 info for current SNP
						hap1 = old_hap1 + cur_hap1
						hap2 = old_hap2 + cur_hap2
						
						# update F matrix
						updated_window = (hap1, hap2)
						F[cur_ind][SNP_window_count] = updated_window
				
	# report results
	print "SNPs analyzed: ", SNP_line_count
	print "number of windows: ", SNP_window_count + 1 	# 0-based
	print "SNPs per window: ", M
	print "individuals: ", len(individuals)

	return individuals, F, SNP_window_count+1

def partition_training(ind_labels, labels):
	label_list = []
	
	with open(labels,"r") as inf:
		for line in inf:
			label_list.append(line.strip())

	M = {} # matrix indexed by pop labels, value = # of individuals
	M = initialize(label_list)
	
	# populate M matrix with individuals indexed according to population label
	with open(ind_labels,"r") as infile:
		for line in infile:
			fields = line.split("\t")
			ind = fields[0]
			lab = fields[1].strip()

			if lab in M:
				M[lab].append(ind)

	# initialize training and test matrices
	training = {}
	test = {}
	tf = 8/10. 		# training_fraction

	for lab in M:
		num_inds = len(M[lab])
		training[lab] = M[lab][0:int(tf*num_inds)]
		test[lab] = M[lab][int(tf*num_inds):]

	return training, test, label_list

def get_admix_data(admixed_inds, F, admixed_label):
	data = {}

	admix_data = {}
	for ind in admixed_inds:
		if ind in F:
			admix_data[ind] = F[ind]

	data[admixed_label] = admix_data

	return data

def get_training(tset, F, label_list):
	data = initialize(label_list)

	for pop in tset:
		# get IDs for individuals in set
		t_inds = tset[pop]

		pop_data = {} 	# matrix indexed by individual IDs, values are SNP windows

		# find SNP data for individuals in F matrix
		for ind in F:
			if ind in t_inds:
				pop_data[ind] = F[ind]

		data[pop] = pop_data

	return data

def initialize_numpy(label_list, training_data):
	for label in training_data:
		cur_pop_data = training_data[label]
		for ind in cur_pop_data:
			hap1 = cur_pop_data[ind][0][0]
			# print "hap1: ", hap1
			tmp_list = list(hap1)
			a = np.array([tmp_list])
			break
		break
	return a

def numpify_and_classify(training_data, test_data, label_list, admixed_label_list, num_windows, test_type):
	# convert data into numpy structure
	X1_train = initialize_numpy(label_list, training_data)
	X2_train = initialize_numpy(label_list, training_data)	
	Y_train= np.array(['init'])

	if test_type == "admixed":
		X1_test = initialize_numpy(admixed_label_list, test_data)
		X2_test = initialize_numpy(admixed_label_list, test_data)
	else:
		X1_test = initialize_numpy(label_list, test_data)
		X2_test = initialize_numpy(label_list, test_data)
		Y_test = np.array(['init'])

	# initialize lists to hold predictions for each haplotype
	hap1_predictions = [] 
	hap2_predictions = []

	for i in xrange(0,num_windows):
		# populate training array
		for label in label_list:
			cur_pop_data = training_data[label]
			for ind in cur_pop_data:
				# add hap1 to numpy array
				hap1 = cur_pop_data[ind][i][0]
				tmp_list = list(hap1)
				tmp = np.array([tmp_list])
				X1_train = np.concatenate((X1_train, tmp))

				# add hap2 to numpy array
				hap2 = cur_pop_data[ind][i][1]
				tmp_list = list(hap2)
				tmp = np.array([tmp_list])
				X2_train = np.concatenate((X2_train, tmp))

				tmp = np.array([label])
				Y_train = np.concatenate((Y_train, tmp))

		if test_type != "admixed":
			# populate tset array
			for label in label_list:
				cur_pop_data = test_data[label]
				for ind in cur_pop_data:
					# add hap1 to numpy array
					hap1 = cur_pop_data[ind][i][0]
					tmp_list = list(hap1)
					tmp = np.array([tmp_list])
					X1_test = np.concatenate((X1_test, tmp))

					# add hap2 to numpy array
					hap2 = cur_pop_data[ind][i][1]
					tmp_list = list(hap2)
					tmp = np.array([tmp_list])
					X2_test = np.concatenate((X2_test, tmp))

					tmp = np.array([label])
					Y_test = np.concatenate((Y_test, tmp))
		else:
			for label in admixed_label_list:
				cur_pop_data = test_data[label]
				for ind in cur_pop_data:
					# add hap1 to numpy array
					hap1 = cur_pop_data[ind][i][0]
					tmp_list = list(hap1)
					tmp = np.array([tmp_list])
					X1_test = np.concatenate((X1_test, tmp))

					# add hap2 to numpy array
					hap2 = cur_pop_data[ind][i][1]
					tmp_list = list(hap2)
					tmp = np.array([tmp_list])
					X2_test = np.concatenate((X2_test, tmp))
					Y_test = None

		# clean up numpy arrays
		X1_train = np.delete(X1_train,0,0)
		X2_train = np.delete(X2_train,0,0)
		Y_train = np.delete(Y_train,0,0)
		
		if test_type != "admixed":
			X1_test = np.delete(X1_test,0,0)
			X2_test = np.delete(X2_test,0,0)
			Y_test = np.delete(Y_test,0,0)

		# classify using SVM, one SNP window and haplotype at a time
		print "--- starting SVM for window ", i+1, ", hap1"
		predicted_labels_h1 = sk_svm(X1_train, Y_train, X1_test, Y_test, label_list, test_type)
		print "--- starting SVM for window ", i+1, ", hap2"
		predicted_labels_h2 = sk_svm(X2_train, Y_train, X2_test, Y_test, label_list, test_type)
		print "done with window ", i+1,"\n----------------------"

		# update predictions for each haplotype
		hap1_predictions.append(predicted_labels_h1)
		hap2_predictions.append(predicted_labels_h2)

		# reset in preparation for next window
		X1_train = initialize_numpy(label_list, training_data)
		X2_train = initialize_numpy(label_list, training_data)	
		Y_train= np.array(['init'])

	return (hap1_predictions, hap2_predictions, Y_test)

def string_kernel(X, Y):
	"""
	We create a custom kernel:

	k(X, Y) = weighted degree string kernel (Ratsch et al, 2006)

	output: N x N matrix, where N is the number of examples (rows)
	"""

	#define kernel_array = np.array of size NxN
	N = X.shape[0] # number of examples/rows
	num_features = X.shape[1] # number of features
	kernel_matrix = np.zeros((N, N))

	for i in range(N):
		x_i = X[i,:] # row i
		for j in range(N):
			x_j = Y[j,:]
			dp_table = []
			# initialize 0 index to 0
			dp_table.append(0) 

			for k in xrange(num_features):
				if x_i[k] == x_j[k]:
					dp_table.append(dp_table[k-1] + 1)
				else:
					dp_table.append(0)
			# finished populating dp_table
			kernel_matrix[i, j] = sum(dp_table)

	return kernel_matrix

""" helper function to calculate length of leading zeros"""
def leading_zeros(change_bit, word_len):
	bit_list = []
	for i in range(word_len):
		bit_list.append(change_bit & 1)
		change_bit >>= 1

	# reverse bit_list
	bit_list[::-1]
	print "bit_list=", bit_list
	try: 
		first_one_pos = bit_list.index(1)
	except ValueError:
		first_one_pos = -1
	return first_one_pos

""" bitvector string kernel function """
def Distance(x, y):
	# x = "010011"
	# y = "010101"
	score_run = []
	word_len = len(x)
	score = 0
	run_len = 0

	if x == y:
		print "identical"
		return comb(word_len, 2)

	print "word_len=", word_len
	# initialization
	for run_len in range(word_len):
		score_run.append(comb(run_len, 2))

	# score_run.pop(0)
	print "score_run= ", score_run

	num_x = int(x, 2)
	num_y = int(y, 2)

	diff = num_x ^ num_y
	diff_shifted = diff << 1
	change_bit = diff ^ diff_shifted

	print "change_bit=", change_bit

	for i in range(word_len):
		print "score=", score
		run_len = leading_zeros(change_bit, word_len)
		print "run_len= ", run_len
		if run_len == -1:
			break
		# print "run_len= ", run_len
		# print "change_bit=", change_bit
		print "score_run[run_len]=", score_run[run_len]
		score += score_run[run_len]
		change_bit <<= (run_len)
	
	return score

def sk_svm(X_train,y_train,X_test, y_test,label_list, test_type):
	"""
	SVM for each segment/window
	
	parameters:
		X_train: training features
		y_train: training labels
		X_test: testing features
		y_test: testing labels
		label_list: list of all population labels
		admixed_bool: True if X_test is admixture testing set --> no y_test
		test_type:	training 	--> training set is test set
					testing		--> testing with known labels
					admixed 	--> predicting admixture
	"""

	# all labels
	all_labels = np.unique(y_train)

	# create and train SVM
	clf = svm.SVC(kernel="linear")				# linear kernel SVM
	# clf = svm.SVC(kernel=string_kernel)		# string kernel SVM
	clf.fit(X_train, y_train)

	# prediction
	if test_type == "training":
		X = X_train
		y = y_train
	else:
		X = X_test
		if test_type == "testing":
			y = y_test#[:X_test.shape[0]]

	# print "x train shape: ", X_train.shape
	# print "x test shape: ", X.shape
	# print "y train shape: ", y_train.shape
	# print "y test shape: ", y.shape
	label_prediction = clf.predict(X)
	# print label_prediction

	if test_type == "admixed":
		return label_prediction

	# ---------- METRICS --------------------

	# cross validation
	print "~~~ cross val ~~~~~"
	scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
	print scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	print "~~ end cross val ~~~"

	# error
	mistakes = np.sum(y != label_prediction)
	error = float(mistakes) / y.shape[0]
	print "error = ", error

	pr = precision_recall_fscore_support(y,label_prediction)

	print "label\tprecision\taccuracy\tF-score"

	# accuracy /recall
	for i, label in enumerate(all_labels):
		predicted_indices = np.where(label_prediction == label)
		actual_indices = np.where(y == label)
		correct_labels = np.intersect1d(predicted_indices,actual_indices)
		all_predicted = np.sum(label_prediction == label)
		actual_label = np.sum(y == label)

		if actual_label == 0:
			accuracy = 0
		else:
			accuracy = float(correct_labels.shape[0]) / actual_label

		# precision 
		if all_predicted == 0:
			precision = 0.0
		else:
			precision = float(correct_labels.shape[0]) / all_predicted

		print label, "\t", precision, "\t", accuracy, "\t", pr[2][i]

	return label_prediction

def initialize(ind_list):
	F = {}
	for ind in ind_list:
		F[ind] = []

	return F

def admixed_inds_to_list(admixed_inds):
	"""
	store IDs of admixed individuals into a list
	"""
	inds = []
	with open(admixed_inds,"r") as infile:
		for line in infile:
			fields = line.split("\t")
			ind = fields[0].strip()
			admixed_label = fields[1].strip()
			inds.append(ind)
	return inds, admixed_label

def output(hap1,hap2,y):
	"""
	output haplotype predictions into .txt file to be used for hmm.py

	format: n rows (number of individuals) x 2 (haplotypes)
		hap1label1,hap1label2,hap1label3	hap2label1,hap2label2,hap2label3
		etc.
	"""
	print "\n\n----> outputting results to predictions.txt"
	print "# windows: ", len(hap1)
	print "# inds: ", len(hap1[0])

	with open('predictions.txt',"w") as outf:
		for i in xrange(0,len(hap1[0])):
			for window in xrange(0,len(hap1)):
				cur_label1 = hap1[window][i]
				cur_label2 = hap2[window][i]
				if window == 0:
					hap1_out = cur_label1
					hap2_out = cur_label2
					# hap1_out = y[i]
					# hap2_out = y[i]
				else:
					hap1_out = hap1_out + "," + cur_label1
					hap2_out = hap2_out + "," + cur_label2
					# hap1_out += "," + y[i]
					# hap2_out += "," + y[i]

			outline = hap1_out + "\t" + hap2_out + "\n"
			#outline = hap1_out + "\t" + hap2_out +  "\t" + y[i] + "\n"
			outf.write(outline)
			
			hap1_out = ''
			hap2_out = ''
			
	return

if __name__ == '__main__':
	# usage: python svm.py chr.vcf labels.txt ind_labelled.txt -t test | train admix [admixed_inds.txt]
	if len(sys.argv) != 6 and len(sys.argv) != 7:
		print "\terror: please check inputs\n\tusage: python svm.py chr.vcf labels.txt ind_labelled.txt -t test | train admix [admixed_inds.txt]"
		sys.exit(1)

	if sys.argv[5] == "admix":
		admixed_bool = True	
		if len(sys.argv) != 7:
			print "\terror: please check inputs\n\tusage: python svm.py chr.vcf labels.txt ind_labelled.txt -t test | train admix [admixed_inds.txt]"
			print "please provide list of admix individuals IDs"
			sys.exit(1)
		test_type = "admixed"
		admixed_inds = sys.argv[6]
		print "starting Ancestry Composition: admixed prediction"

	else:
		admixed_bool = False
		if sys.argv[5] == "test":
			test_type = "testing"		# get test error
			print "starting Ancestry Composition: testing"

		else:
			test_type = "training"		# get training error
			print "starting Ancestry Composition: training"

	chr_vcf = sys.argv[1]
	labels = sys.argv[2]
	ind_labels = sys.argv[3]

	# step 1: pre-process VCF into windows
	inds, F, num_windows = process_vcf(chr_vcf)

	# step 2: partition individuals into training and test sets
	train_set, test_set, label_list = partition_training(ind_labels, labels)

	# step 3: get SNP windows for training and test sets
	training_data = get_training(train_set, F, label_list)
	
	if admixed_bool == False:
		test_data = get_training(test_set, F, label_list)
		admixed_label_list = None
	else:
		# test admixed individuals
		admix_ind_list, admixed_label = admixed_inds_to_list(admixed_inds)
		test_data = get_admix_data(admix_ind_list, F, admixed_label)
		admixed_label_list = [admixed_label]

	# step 4: convert data into numpy structure
	hap1_predictions, hap2_predictions, y_test = numpify_and_classify(training_data, test_data, label_list, admixed_label_list, num_windows, test_type)
	
	# print "predictions"
	# print hap1_predictions, hap2_predictions
	# step 5: output predictions from SVM
	output(hap1_predictions, hap2_predictions, y_test)

	print "done."
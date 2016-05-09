###
# hmm.py
# COMS 4761 Project: 23and1000Genomes 
# Katie Lin, Joy Pai
###

import sys
import numpy as np

def HMM_main(x_labels, labels):
	'''
		main function to build HMM and test 

	'''

	labels_as_nums = create_labels_as_nums(labels)	# create label index dict
	nums_as_labels = create_nums_as_labels(labels_as_nums) # create number index dict

	mu_k, mu_k_y, sigma, theta, e_y_x = initialize_probs(labels)

	num_segments = len(x_labels[0][0])		# number of windows/segments
	num_pops = len(mu_k)				# number of population labels = 20
	numstates = num_pops * num_pops

	sigma, theta, eps, mu_k, mu_k_y = HMM(x_labels, mu_k, mu_k_y, sigma, theta, e_y_x, labels, labels_as_nums, nums_as_labels)
	sys.stderr.write("finished HMM training\n")

	# for test in x_labels:
	test_label = [x_labels[24]]
	print "test_label = ", test_label

	fwd = forward_probs(num_segments, num_pops, test_label, mu_k, mu_k_y, sigma, theta, e_y_x, labels, labels_as_nums)
	bkd = backward_probs(num_segments, num_pops, test_label, mu_k, mu_k_y, sigma, theta, e_y_x, labels, labels_as_nums)

	sys.stderr.write("finished fwd-bkd on test\n")
	# combine it
	prob_mat = np.array(fwd)*np.array(bkd)

	# normalize
	for row in range(0, prob_mat.shape[0]):
		sum_row = prob_mat[row].sum() 
		prob_mat[row] = prob_mat[row] / sum_row

	sequence, prob = viterbi(prob_mat, nums_as_labels)

	print "sequence=", sequence
	print "prob=", prob

	return

def HMM(x_labels, mu_k, mu_k_y, sigma, theta, e_y_x, labels, labels_as_nums, nums_as_labels):
	'''
		Update HMM parameters using forward-backward algorithm and expectation-maximization
			x_labels: predicted labels from SVM
			mu_k: prior prob for hidden states
			mu_k_y: prior prob for emissions conditional on hidden state y
			sigma: prior prob of switch 
			theta: prior prob of recombination between two consecutive positions
			e_y_x: prior probability of label reset 
	'''

	num_segments = len(x_labels[0][0])		# number of windows/segments
	num_pops = len(mu_k)				# number of population labels = 20
	numstates = num_pops * num_pops

	for i in range(5):
		fwd = forward_probs(num_segments, num_pops, x_labels, mu_k, mu_k_y, sigma, theta, e_y_x, labels, labels_as_nums)
		bkd = backward_probs(num_segments, num_pops, x_labels, mu_k, mu_k_y, sigma, theta, e_y_x, labels, labels_as_nums)
		gamma, xi = expectation(x_labels, num_segments, numstates, fwd, bkd, nums_as_labels, e_y_x, theta, mu_k, mu_k_y)
		sigma, theta, e_y_x, mu_k, mu_k_y = maximization(x_labels, labels, num_segments, numstates, fwd, nums_as_labels, xi, gamma)
		sys.stderr.write("HMM iteration %s\n" % i)
	return sigma, theta, e_y_x, mu_k, mu_k_y 

def forward_probs(num_segments, num_pops, x_labels, mu_k, mu_k_y, sigma, theta, e_y_x, labels, labels_as_nums):
	'''
		forward algorithm
	'''
	# initialization
	fwd_global = np.zeros([num_segments, num_pops*num_pops])		# initialize fwd matrix to zeros

	for individual in x_labels:
		fwd_individual = np.zeros([num_segments, num_pops*num_pops]) # reset individual
		h0 = individual[0]
		h1 = individual[1]

		# initial distribution
		for label_h0 in labels:
			for label_h1 in labels:
				
				p_y1 = mu_k[label_h0] * mu_k[label_h1] # P(y1)
				p_y1x1 = mu_k_y[(h0[0], label_h0)] * mu_k_y[(h1[0], label_h1)] # P(x1|y1)

				fwd_individual[0][labels_as_nums[(label_h0,label_h1)]] = np.logaddexp(p_y1, p_y1x1)
				fwd_global[0][labels_as_nums[(label_h0,label_h1)]] += np.logaddexp(p_y1, p_y1x1)

		# windows 2 - S
		for i in xrange(1, len(h0)):
			if h0[i-1] == h1[i-1]:
				# unknown if switch happened
				st = 0
			else: 
				if h0[i-1] == h1[i] and h1[i-1] == h0[i]:
					st = 1
				else:
					st = 0

			for label_minus1_h0 in labels:
				for label_h0 in labels:
					for label_minus1_h1 in labels:
						for label_h1 in labels:
							p_switch = switch_error(sigma, st) 
							p_transition = transition_prob(label_h0, label_minus1_h0, label_h1, label_minus1_h1, st, theta, mu_k)
							p_emission = emission_prob(h0[i], h0[i-1], label_h0, label_minus1_h0, h1[i], h1[i-1], label_h1, label_minus1_h1, st, e_y_x, mu_k_y)
							
							# logadd = np.logaddexp(p_transition, p_emission)
							prob_i = np.logaddexp(p_switch, np.logaddexp(p_transition, p_emission))
							fwd_individual[i][labels_as_nums[(label_h0, label_h1)]] = prob_i
							fwd_global[i][labels_as_nums[(label_h0, label_h1)]] += prob_i
	# normalize
	for row in range(0, fwd_global.shape[0]):
		sum_row = fwd_global[row].sum() 
		fwd_global[row] = fwd_global[row] / sum_row
	
	return fwd_global

def backward_probs(num_segments, num_pops, x_labels, mu_k, mu_k_y, sigma, theta, e_y_x, labels, labels_as_nums):
	'''
		forward algorithm
	'''
	# initialization
	
	bkd_individual = np.zeros([num_segments, num_pops*num_pops])		# initialize bkd matrix to zeros
	bkd_global = np.zeros([num_segments, num_pops*num_pops])		# initialize bkd matrix to zeros

	for individual in x_labels:
		bkd_individual = np.zeros([num_segments, num_pops*num_pops]) # reset individual
		h0 = individual[0]
		h1 = individual[1]

		# initial distribution
		for label_h0 in labels:
			for label_h1 in labels:
				# P(y1)
				p_y1 = mu_k[label_h0] * mu_k[label_h1]
				# P(x1|y1)
				p_y1x1 = mu_k_y[(h0[len(h0)-1], label_h0)] * mu_k_y[(h1[len(h0)-1], label_h1)]
				bkd_individual[len(h0)-1][labels_as_nums[(label_h0,label_h1)]] = np.logaddexp(p_y1, p_y1x1)
				bkd_global[len(h0)-1][labels_as_nums[(label_h0,label_h1)]] += np.logaddexp(p_y1, p_y1x1)

		# windows 2 - S
		for i in xrange(len(h0)-2, -1,-1):
			if h0[i+1] == h1[i+1]:
				# unknown if switch happened
				st = 0
			else: 
				if h0[i+1] == h1[i] and h1[i+1] == h0[i]:
					st = 1
				else:
					st = 0

			for label_minus1_h0 in labels:
				for label_h0 in labels:
					for label_minus1_h1 in labels:
						for label_h1 in labels:
							p_switch = switch_error(sigma, st) 
							p_transition = transition_prob(label_h0, label_minus1_h0, label_h1, label_minus1_h1, st, theta, mu_k)
							p_emission = emission_prob(h0[i], h0[i+1], label_h0, label_minus1_h0, h1[i], h1[i+1], label_h1, label_minus1_h1, st, e_y_x, mu_k_y)
							
							# logadd = np.logaddexp(p_transition, p_emission)
							prob_i = np.logaddexp(p_switch, np.logaddexp(p_transition, p_emission))
							bkd_individual[i][labels_as_nums[(label_h0, label_h1)]] = prob_i
							bkd_global[i][labels_as_nums[(label_h0, label_h1)]] += prob_i
	# normalize
	for row in range(0, bkd_global.shape[0]):
		sum_row = bkd_global[row].sum() 
		bkd_global[row] = bkd_global[row] / sum_row
	return bkd_global

def expectation(x_labels, num_segments, numstates, forward, backward, nums_as_labels, e_y_x, theta, mu_k, mu_k_y):
	'''
		expectation step:
		re-estimate xi_t(i, j) and gamma_t(j)
		returns two things:
		- gamma is a (N, T) numpy matrix
		- xi is a list of T numpy matrices of size (N, N)
	'''
	# computing xi
	xi = [ ]
	s_t_total = 0.0
	recomb_count = 0.0
	total = 0.0
	eps = 0.0

	for seg_window in range(num_segments - 1):
		xi_t = np.zeros((numstates, numstates))
		
		for s_i in range(numstates): # t-1
			for s_j in range(numstates): # t            	
				y_t_h0, y_t_h1 = nums_as_labels[s_j]
				y_t_1_h0, y_t_1_h1 = nums_as_labels[s_i]

				# theta
				if y_t_h1 != y_t_1_h1:
					recomb_count += 1
				if y_t_h0 != y_t_1_h0:
					recomb_count += 1

				if y_t_1_h0 == y_t_1_h1:
					# unknown if switch happened
					s_t = 0
					total += 1
				else: 
					if y_t_1_h0 == y_t_h1 and y_t_1_h1 == y_t_h0:
						s_t = 1
						s_t_total += 1
						total += 1
					else:
						s_t = 0
						total += 1

				trans_prob = transition_prob(y_t_h0, y_t_1_h0, y_t_h1, y_t_1_h1, s_t, theta, mu_k)

				em_prob = 0
				for individual in x_labels:
					# individual = ( (x1,x2,x3...),(x1, x2, x3...) )
					x_t_h0 = individual[0][seg_window + 1]
					x_t_h1 = individual[1][seg_window + 1]
					x_t_1_h0 = individual[0][seg_window]
					x_t_1_h1 = individual[1][seg_window]

					if x_t_1_h0 == y_t_1_h0 and y_t_1_h0 == y_t_h0:
						if x_t_h0 != y_t_h0:
							eps += 1

					em_prob += emission_prob(x_t_h0, x_t_1_h0, y_t_h0, y_t_1_h0,x_t_h1, x_t_1_h1, y_t_h1, y_t_1_h1, s_t, e_y_x, mu_k_y)

				trans_em_log = np.logaddexp(trans_prob, em_prob)
				xi_t[ s_i, s_j] = trans_em_log + forward[seg_window, s_i] + backward[seg_window+1, s_j]

		xi.append(xi_t)

	# computing gamma
	gamma = np.zeros((numstates, num_segments))
	for t in range(num_segments - 1):
		for s_i in range(numstates):
			gamma[s_i, t] = sum([ xi[t][s_i, s_j] for s_j in range(numstates) ])

		for s_j in range(numstates):
			gamma[s_j, num_segments - 1] = sum( [ xi[t][s_i, s_j] for s_i in range(numstates) ] )
			
	
	return (gamma, xi)

def maximization(x_labels, labels, num_segments, numstates, forward, nums_as_labels, xi, gamma):
	'''
		maximization step:
		re-estimate trans, emis based on gamma, xi
		returns:
		- initialprob
		- trans
		- emis
	'''
	s_t_total = 0.0
	recomb_count = 0.0
	total = 0.0
	mu_k = {} # prior distribution for K hidden states
	mu_k_total = 0.0
	mu_e = {} 
	mu_e_total = 0.0
	mu_k_y = {} # prior distribution for hidden state y
	mu_k_y_total = 0.0
	e_y_x = {} # prior probability of a label reset between two consecutive positions with the same hidden state y, where the former label is x

	for label in labels:
		mu_k[label] = 0
		mu_e[label] = 0
		for label2 in labels:
			mu_k_y[(label, label2)] = 0
			e_y_x[(label, label2)] = 0

	# normalize for t
	for row in range(0, len(xi)):
		sum_row = xi[row].sum() 
		xi[row] = xi[row] / sum_row

	# normalize gamma, sum of each column = 1
	for col in range(0, gamma.shape[1]):
		sum_column = gamma[:,col].sum() 
		gamma[:,col] = gamma[:,col] / sum_column

	for t in range(num_segments - 1):
		xi_t = xi[t]
		# normalize for t
		for row in range(0, len(xi_t)):
			sum_row = xi_t[row].sum() 
			xi_t[row] = xi_t[row] / sum_row

		for i in range(numstates):
			s_i = nums_as_labels[i]
			for j in range(numstates):
				s_j = nums_as_labels[j]
				total += 1
				if s_i[0] == s_j[1] and s_i[1] == s_j[0]:
					s_t_total += xi_t[i, j]
				if s_i[0] != s_j[0]:
					recomb_count += xi_t[i, j]
				if s_i[1] != s_j[1]:
					recomb_count += xi_t[i, j]

				# reset: eps
				if s_i[0] == s_j[0] == s_i[1] == s_j[1]:
					gamma_t = gamma[:, t]
					gamma_t_p1 = gamma[:, t + 1]
					for gamma_i in range(gamma_t.shape[0]):
						prob_x_t = gamma_t[gamma_i]
						for gamma_j in range(gamma_t.shape[0]):
							if gamma_i != gamma_j:
								prob_x_t_p1 = gamma_t_p1[gamma_j]
								label_x_prime = nums_as_labels[gamma_i]
								e_y_x[s_i[0], label_x_prime[0]] += prob_x_t + prob_x_t_p1

			# prior distribution of hidden states, mu_e
			for label in labels:
				if s_i[0] == label:
					mu_e[label] += gamma[i][t]
					mu_e_total += gamma[i][t]

					mu_k[label] += forward[t][i]
					mu_k_total += forward[t][i]

				if s_i[1] == label:
					mu_e[label] += gamma[i][t]
					mu_e_total += gamma[i][t]

					mu_k[label] += forward[t][i]
					mu_k_total += forward[t][i]

				if s_j[0] == label:
					mu_e[label] += gamma[j][t]
					mu_e_total += gamma[j][t]

					mu_k[label] += forward[t][j]
					mu_k_total += forward[t][j]

				if s_j[1] == label:
					mu_e[label] += gamma[j][t]
					mu_e_total += gamma[j][t]

					mu_k[label] += forward[t][j]
					mu_k_total += forward[t][j]

	sigma = s_t_total / total
	theta = recomb_count / total

	for label in mu_e:
		mu_e[label] = mu_e[label] / mu_e_total
		mu_k[label] = mu_k[label] / mu_k_total

	for label in labels:
		for label2 in labels:
			mu_k_y[(label, label2)] = (mu_e[label] * mu_k[label2]) / mu_k[label2]

	return (sigma, theta, e_y_x, mu_k, mu_k_y)


def transition_prob(y_t_h0, y_t_1_h0, y_t_h1, y_t_1_h1, s_t, theta, mu_k):
	'''
		transition probability
	'''
	same_state_h0 = bool(0) ^ bool(s_t)
	if same_state_h0:
		t = f(y_t_h0, (y_t_h0 == y_t_1_h0), theta, mu_k)
	else:
		t = f(y_t_h0, (y_t_h0 == y_t_1_h1), theta, mu_k)

	same_state_h1 = bool(1) ^ bool(s_t)
	if same_state_h1:
		t = t * f(y_t_h1, (y_t_h1 == y_t_1_h1), theta, mu_k)
	else:
		t = t * f(y_t_h1, (y_t_h1 == y_t_1_h0), theta, mu_k)

	return t

def f(y, same_state, theta, mu_k):
	'''
		helper function for transition probability
	'''
	return theta * mu_k[y] + ((1 - theta) * same_state)


def emission_prob(x_t_h0, x_t_1_h0, y_t_h0, y_t_1_h0, x_t_h1, x_t_1_h1, y_t_h1, y_t_1_h1, s_t, e_y_x, mu_k_y):
	'''
		emission probability
	'''
	same_state_h0 = bool(0) ^ bool(s_t)
	if same_state_h0:
		e = g(x_t_h0, x_t_1_h0, y_t_h0, (y_t_h0 == y_t_1_h0), e_y_x, mu_k_y)
	else:
		e = g(x_t_h0, x_t_1_h1, y_t_h0, (y_t_h0 == y_t_1_h1), e_y_x, mu_k_y)

	same_state_h1 = bool(1) ^ bool(s_t)
	if same_state_h1:
		e = e * g(x_t_h1, x_t_1_h1, y_t_h1, (y_t_h1 == y_t_1_h1), e_y_x, mu_k_y)
	else:
		e = e * g(x_t_h1, x_t_1_h0, y_t_h1, (y_t_h1 == y_t_1_h0), e_y_x, mu_k_y)
	return e

def g(x, x_prime, y, same_state, e_y_x, mu_k_y):
	'''
		helper function for emission probability
	'''
	if x == x_prime:
		same_prev_x = 1
	else:
		same_prev_x = 0

	if same_state:
		return e_y_x[y, x_prime] * mu_k_y[x, y] + (1 - e_y_x[y, x_prime]) * same_prev_x
	else:
		return mu_k_y[x, y]

def initialize_probs(labels):
	'''
		Initialize free variables as a uniform distribution 
	'''
	k = len(labels)

	mu_k = {} # prior distribution for K hidden states
	mu_k_y = {} # prior distribution for hidden state y
	e_y_x = {} # prior probability of a label reset between two consecutive positions with the same hidden state y, where the former label is x

	for label in labels:
		mu_k[label] = 1./k

		for label2 in labels:
			mu_k_y[(label, label2)] = 1./(k*k)
			e_y_x[(label, label2)] = 1./(k*k)

	sigma = 0.5		# arbitrary
	theta = 0.1 	# arbitrary

	return mu_k, mu_k_y, sigma, theta, e_y_x

def switch_error(sigma, st):
	'''
		st = 0 or 1 (switch indicator)
		sigma = prior probability of switch

		returns switch error model probability
	'''

	if st == 1:
		return sigma
	elif st == 0:
		return (1-sigma)
	else:
		print "something wrong: st not 0 or 1"
		return

def viterbi(prob_mat, nums_as_labels):
	'''
		Viterbi algorithm
	'''
	max_in_each_row = np.argmax(prob_mat, axis=1)
	prob = []
	sequence = []
	for row_i in range(prob_mat.shape[0]):
		max_col = max_in_each_row[row_i]
		p = prob_mat[row_i][max_col]
		prob.append(p)
		label = nums_as_labels[max_col]
		sequence.append(label)

	return sequence, prob

def create_labels_as_nums(labels):
	'''
		returns mapping from population labels to numbers
	'''
	labels_as_nums = {}

	i = 0
	for label_h0 in labels:
		for label_h1 in labels:
			labels_as_nums[(label_h0, label_h1)] = i
			i = i+1

	return labels_as_nums

def create_nums_as_labels(labels_as_nums):
	'''
		returns mapping from numbers to population labels
	'''
	nums_as_labels = {}

	for label, num in labels_as_nums.iteritems():
		nums_as_labels[num] = label

	return nums_as_labels

def hap_file_to_list(file_name):
	'''
		Given a file where each line is formatted as 'x1, x2, x3...xn \t x1, x2, x3...xn',
		a list representation of the observed states is returned
			x_labels: [((x1,x2,x3...),(x1, x2, x3...)), 
				( (x1,x2,x3...),(x1, x2, x3...) ), 
				( (x1,x2,x3...),(x1, x2, x3...) )]
	'''
	x_labels = []
	with open(file_name) as f:
		for line in f:
			# line = x1, x2, x3...xn \t x1, x2, x3...xn
			hap0, hap1 = line.strip().split('\t')
			hap0 = tuple(hap0.split(","))
			hap1 = tuple(hap1.split(","))
			x_labels.append((hap0, hap1))
	return x_labels

def labels_file_to_list(file_name):
	'''
		Given a file with a list of labels, 
		a list representation of the labels is returned
	'''
	labels_list = []
	with open(file_name,"r") as inf:
		for line in inf:
			labels_list.append(line.strip())
	return labels_list


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print "\terror: not enough inputs\n\tusage: python predictions.txt labels.txt"
		sys.exit(1)
	x_labels = hap_file_to_list(sys.argv[1])
	labels = labels_file_to_list(sys.argv[2])
	HMM_main(x_labels, labels)

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

"""
hmm_output: from hmm.py
- format
	prob_window1	prob_window2	prob_window3	... 
"""
def get_probs(hmm_output):
	with open(hmm_output,"r") as infile:
		# read probabilities from file into numpy array
		for line in infile:
			fields = line.split(",")
	
			probabilities = np.zeros([len(fields),])

			for i in range(0,len(fields)):
				probability = float(fields[i].strip())
				probabilities[i] = probability

	return probabilities


""" fit isotonic regression model for each individual's probabilities
	to recalibrate confidence scores

	outputs graph of isotonic and linear regression models
"""
def ir(assignment_probs):
	num_segments = assignment_probs.shape[0]
	print "num segs: " , num_segments
	x = np.arange(num_segments)
	y = assignment_probs

	print "x: ", x
	print "y: ", y

	# fit isotonic and linear regression models
	ir = IsotonicRegression()
	y_ = ir.fit_transform(x, y)
	lr = LinearRegression()
	lr.fit(x[:, np.newaxis], y)

	segments = [[[i, y[i]], [i, y_[i]]] for i in range(num_segments)]
	lc = LineCollection(segments, zorder=0)
	lc.set_array(np.ones(len(y)))
	lc.set_linewidths(0.5 * np.ones(num_segments))

	# plot models
	fig = plt.figure()
	plt.plot(x, y, 'r.', markersize=12)
	plt.plot(x, y_, 'g.-', markersize=12)
	plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
	plt.gca().add_collection(lc)
	plt.legend(('HMM probabilities', 'Isotonic Fit', 'Linear Fit'), loc='upper left')
	plt.show()

	return

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print "\terror: not enough inputs\n\tusage: python ir.py prob_file"
		sys.exit(1)

	assignment_probs = get_probs(sys.argv[1])
	ir(assignment_probs)
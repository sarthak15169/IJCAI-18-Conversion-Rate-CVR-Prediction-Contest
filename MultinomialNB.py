from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
import numpy as np
X, y = np.loadtxt('samples.out'), np.loadtxt('lables.out')
print "X.shape: ", X.shape, ", y.shape: ", y.shape

print "Naive Bayes: ", cross_val_score(GaussianNB(), X, y, cv=5).mean()


from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
import numpy as np
X, y = np.loadtxt('samples.out'), np.loadtxt('lables.out')
print "X.shape: ", X.shape, ", y.shape: ", y.shape

print "SVM: ", cross_val_score(svm.NuSVC(), X, y, cv=5).mean()

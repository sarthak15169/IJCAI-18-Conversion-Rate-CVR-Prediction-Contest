from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
import numpy as np
X, y = np.loadtxt('samples.out'), np.loadtxt('lables.out')
print "X.shape: ", X.shape, ", y.shape: ", y.shape

print "Neural Net with Bagging: ", cross_val_score(BaggingClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(1)), n_estimators=50), X, y, cv=5).mean()

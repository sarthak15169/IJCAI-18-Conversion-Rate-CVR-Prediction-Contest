from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
X, y = np.loadtxt('samples.out'), np.loadtxt('lables.out')
print "X.shape: ", X.shape, ", y.shape: ", y.shape

print "Neural Net: ", cross_val_score(MLPClassifier(hidden_layer_sizes=(512, 256)), X, y, cv=5).mean()

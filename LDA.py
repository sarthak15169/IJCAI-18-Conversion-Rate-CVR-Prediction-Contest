from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
X, y = np.loadtxt('samples.out'), np.loadtxt('lables.out')
print "X.shape: ", X.shape, ", y.shape: ", y.shape

print "Naive Bayes: ", cross_val_score(GaussianNB(), LDA(n_components=2).fit_transform(X, y), y, cv=5).mean()


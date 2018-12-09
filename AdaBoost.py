from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
X, y = np.loadtxt('samples.out'), np.loadtxt('lables.out')
print "X.shape: ", X.shape, ", y.shape: ", y.shape

print "Naive Bayes with AdaBoost: ", cross_val_score(AdaBoostClassifier(base_estimator=MultinomialNB(), n_estimators=50), X, y, cv=5).mean()

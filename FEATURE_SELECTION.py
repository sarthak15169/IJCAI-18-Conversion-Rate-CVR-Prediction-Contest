from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import numpy as np
X, y = np.loadtxt('samples.out'), np.loadtxt('lables.out')
for i in range(1, 15):
    print i , " -> Naive Bayes : ", cross_val_score(GaussianNB(), SelectKBest(mutual_info_classif, k=i).fit_transform(X, y), y, cv=5).mean()
    

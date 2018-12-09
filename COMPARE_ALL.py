import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

X, y = np.loadtxt('samples.out'), np.loadtxt('lables.out')
print "X.shape: ", X.shape, ", y.shape: ", y.shape

h = .02  # step size in the mesh

names = ["Gaussian Naive Bayes", 
         "NN (14)",
         "Nearest Neighbors",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
          "QDA"]

classifiers = [
    GaussianNB(),
    MLPClassifier(hidden_layer_sizes=(14)),
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    QuadraticDiscriminantAnalysis()]

for name, clf in zip(names, classifiers):
    print name, ": ", cross_val_score(clf, LDA(n_components=2).fit_transform(X, y), y, cv=StratifiedKFold(n_splits=5, shuffle = True)).mean()

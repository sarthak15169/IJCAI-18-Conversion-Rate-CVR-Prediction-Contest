import matplotlib.pyplot as plt
import pandas as pd
import numpy 

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from pandas.plotting import parallel_coordinates

def read(feature_index):
    fp = open('round1_ijcai_18_train_20180301.txt', 'r')
    feature_names = fp.readline().split()
    samples = []
    labels = []

    #Reading samples
    for i in xrange(100):
        l = fp.readline().split()
        samples.append([l[feature_index]])
        labels.append(l[-1])

    fp.close()

    data = numpy.array(samples).astype(numpy.float)
    target = numpy.array(labels).astype(numpy.float)
    return data, target, feature_names[feature_index]

#X, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0)
skip = [2, 3, 18]
for i in xrange(26):
    if i in skip:
        continue
    X, y, feature_name = read(i)
    print "Deciding on feature "  + str(i) + ": "+ feature_name
    X_norm = (X - X.min())/(X.max() - X.min())
    plt.scatter(X_norm, y, c=y)
    plt.xlabel(feature_name)
    plt.ylabel('Converted or Not')
    plt.title('Deciding whether to select the feature ' + feature_name)
    #plt.show()
    plt.savefig('C:/Users/Sarthak Jindal/Downloads/ieee-latex-conference-template/IEEEtran/images/feature_selection/' + feature_name + '.png')

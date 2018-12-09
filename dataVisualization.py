import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from pandas.plotting import parallel_coordinates

#X, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0)
X, y = np.loadtxt('samples.out'), np.loadtxt('lables.out')
#X_norm = (X - X.min())/(X.max() - X.min())
X_norm = X
#plt.scatter(X_norm[:,0], X_norm[:,1], c=y)
#plt.show()

pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))
#plt.scatter(transformed[:,0], transformed[:,1], c=y)
plt.scatter(transformed[y==0][0], transformed[y==0][1], label='Class 0', c='red')
plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Class 1', c='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Principal Component Analysis (n_components = 2)')
plt.legend(loc="lower right")
plt.show()

lda = LDA(n_components=2) #2-dimensional LDA
lda_transformed = pd.DataFrame(lda.fit_transform(X_norm, y))
#print lda_transformed
plt.scatter(lda_transformed[y==0][0], [0] * len(lda_transformed[y==0][0]), label='Class 0', c='red')
plt.scatter(lda_transformed[y==1][0], [0] * len(lda_transformed[y==1][0]), label='Class 1', c='blue')
#plt.scatter(lda_transformed[:,0], [0] * len(lda_transformed[:,0]), c=y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Discriminant Analysis ')
plt.legend(loc="lower right")
plt.show()





import numpy
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = np.loadtxt('samples.out'), np.loadtxt('lables.out')
print "X.shape: ", X.shape, ", y.shape: ", y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = SVC(gamma=2, C=1)
clf.fit(X_train, y_train)
numpy.savetxt('predictions.out', clf.predict(X_test))
numpy.savetxt('actual.out', y_test)
print clf.score(X_test, y_test)

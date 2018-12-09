import numpy
from readData import read
X, y = read()
print X.shape, y.shape
numpy.savetxt('samples.out', X)
numpy.savetxt('lables.out', y)

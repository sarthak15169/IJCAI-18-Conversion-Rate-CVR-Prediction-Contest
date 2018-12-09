from readData import read, fiveFoldCrossValidation

X, y = read()
for i in range(22):
    fiveFoldCrossValidation(X[:, range(i + 1)], y)

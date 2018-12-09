import numpy
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

feature_index = [6, 7, 8, 9, 11, 12, 14, 17, 20, 21, 22, 23, 24, 25]
feature_name = ["item_price_level", "item_sales_level", "item_collected_level", "item_pv_level", "user_gender_id", "user_age_level", "user_star_level", "context_page_id", "shop_review_num_level", "shop_review_positive_rate", "shop_star_level", "shop_score_service", "shop_score_delivery", "shop_score_description"]

def read():
    fp = open('round2_train.txt', 'r')
    line = fp.readline()
    samples, labels = [], []
    numzeroes, numones = 0, 0

    #Reading samples
    while line:
        line = fp.readline()
        l = line.split()
        if (len(l) < len(feature_index)):
            break
        missing = 0
        for index in feature_index:
            if float(l[index]) == -1:
                missing = 1
                break
        if missing == 1:
            continue
            
        if (int(l[-1]) == 0) :
            if numzeroes == numones :
                continue
            else:
                numzeroes += 1
        else :
            numones += 1
            
        sample = []
        for index in feature_index:
            sample.append(float(l[index]))
            
        if sample not in samples:
            samples.append(sample)
            labels.append(int(l[-1]))
    
    fp.close()
    return numpy.array(samples), numpy.array(labels)

def readPca():
    X, y = read()
    X_norm = (X - X.min())/(X.max() - X.min())
    pca = sklearnPCA(n_components=2) #2-dimensional PCA
    transformed = (pca.fit_transform(X_norm))
    return transformed, y

def readLda():
    X, y = read()
    X_norm = (X - X.min())/(X.max() - X.min())
    lda = LDA(n_components=2) #2-dimensional LDA
    lda_transformed = (lda.fit_transform(X_norm, y))
    return lda_transformed, y
    

"""
X, y = readPca()
print "PCA Transform "
fiveFoldCrossValidation(X, y)
X, y = readLda()
print "LDA Transform "
fiveFoldCrossValidation(X, y)

"""


    

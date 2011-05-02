import rpy2.robjects as robjects
rsummary = robjects.r("summary(iris)")
print rsummary
pairs(iris, main="Iris Data (red=setosa,green=versicolor,blue=virginica)", pch=21,
  bg=c("red","green3","blue")[unclass(iris$Species)])
from scikits.learn import datasets
from scikits.learn.cross_val import StratifiedKFold
from scikits.learn import svm
from scikits.learn.linear_model import LogisticRegression
from scikits.learn import neighbors
from rpy2 import robjects
from rpy2.robjects import FloatVector as rfloat
import numpy as np

iris = datasets.load_iris()

def test_algorithm(algorithm, results, train_data, train_target, test_data):
    algorithm.fit(train_data, train_target)
    y_pred = algorithm.predict(test_data)
    results.append(precision(y_pred))

def precision(y_pred):
    prec = sum(y_pred == test_target)
    return float(prec) / len(test_target)

svmclf = svm.SVC()
logisticclf = LogisticRegression()
nnclf= neighbors.Neighbors()
svmli = []
logli = []
nnli = []

cv = StratifiedKFold(iris.target, 20)
for train_index, test_index in cv:
    train_data = iris.data[train_index]
    train_target = iris.target[train_index]
    test_data = iris.data[test_index]
    test_target = iris.target[test_index]

    #svm
    test_algorithm(svmclf, svmli, train_data, train_target, test_data)

    #logistic regression
    test_algorithm(logisticclf, logli, train_data, train_target, test_data)
    
    #NN
    test_algorithm(nnclf, nnli, train_data, train_target, test_data)


print "Precison of each algortihm:"
print "SVM:", np.average(svmli)
print "logistic regression:", np.average(logli)
print "nearest neighbors:", np.average(nnli)
print

rttest = robjects.r["t.test"]
print "Paired t-test between algorithms:"
print "SVM vs logistic regression",
tt =  rttest(rfloat(svmli), rfloat(logli), paired=True)
print "p-value", tt.rx('p.value')[0][0]
print "SVM vs nearest neighbors",
tt = rttest(rfloat(svmli), rfloat(nnli), paired=True)
print "p-value", tt.rx('p.value')[0][0]
print "nearest neighbors vs logistic regression",
tt = rttest(rfloat(logli), rfloat(nnli), paired=True)
print "p-value", tt.rx('p.value')[0][0]

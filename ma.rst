==============================================================
A Comparison of Several Classification Algorithms
==============================================================

:Author: Meng Xinfan<mxf3306@gmail.com>

Introduction
==============
Classification is the task that put data into different categories
based on their characteristics.
For instance, given an book, we can put it into a fiction category
or a science category based on its topic.
Classifier is the computer algorithm that conduct classification task.
There are two types of classification: supervised and unsupervised.
Supervised classifier is the classifier that are given the categories and data in advance, 
from which it can learn how to classify data.
By contrast, unsupervised classifier do not have access to data in advance.


In this report, we are
going to compare the performance of several classification algorithms.
The data we use is the iris dataset.

The Exploration of the Dataset
==============

Iris dataset is a classic dataset used in classification.
It was first used by Sir Ronald Aylmer Fisher, a renowned statistician.
The dataset contains 50 samples from each of three species of Iris flowers (Iris setosa, Iris virginica and Iris versicolor), 
a genus of 260 species of flowering plants with showy flowers.

First, we will do some exploratory data analysis (EDA) with R and Python.
Let us have a look at the summary of iris dataset.

::

  import rpy2.robjects as robjects
  rsummary = robjects.r("summary(iris)")
  print rsummary


::

    Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
   Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
   1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
   Median :5.800   Median :3.000   Median :4.350   Median :1.300  
   Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
   3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
   Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  
         Species  
   setosa    :50  
   versicolor:50  
   virginica :50  
                  
                  
                  
  



 From the summary, we know each instance in the dataset have Four features: 
sepal length sepal width, petal length and petal width
(species is not considered a feature since that is what we would like to predict).

Then we draw a scatter plot of the dataset.
::

  pairs(iris, main="Iris Data (red=setosa,green=versicolor,blue=virginica)", pch=21,
    bg=c("red","green3","blue")[unclass(iris$Species)])


 
From this graph, we can see that most features have considerable power in 
discriminating species.


A Comparison of Classifier Algorithms
================================
::

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


::

  Precison of each algortihm:
  SVM: 0.927777777778
  logistic regression: 0.941666666667
  nearest neighbors: 0.958333333333
  
  Paired t-test between algorithms:
  SVM vs logistic regression p-value 0.715682107694
  SVM vs nearest neighbors p-value 0.185647078924
  nearest neighbors vs logistic regression p-value 0.540884257545




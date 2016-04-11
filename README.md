# Classification_FeatureSelect_GUI
author: Tzu-Ching Wu(George)

![alt tag](https://github.com/George-wu509/Kaggle_Customer-GUI/blob/master/figure1.png)


This project contains the source code of Kaggle_Customer GUI, a matlab-based GUI 
for the data preprocessing, feature selection, and classification method testing
of the Kaggle competition: Santander Customer Satisfaction. You will obtain the 
averaged AUC value to evaluate classification performance after running, and 
can easily create the test dataset result csv file for submitting.


Requirement
-------------------------
Two matlab-GUI related code: Customer_GUI.m and Customer_GUI.fig are provided. 
The subfolder /pre_data shoud contains three matlab data file: train.mat, test.mat, 
and ID.mat which are training dataset, testing dataset, and testing ID from Santander 
Customer Satisfaction website. Of course, you should have matlab installed in your compyter.  


Running Kaggle_Customer-GUI
-------------------------
To run Kaggle_Customer-GUI, just type Customer_GUI in matlab command line.

1.Pre-data method:
* Raw data
* Normalization
* Binary

2.Feature selection
* All features
* Choice feature, and input feature numbers

3.Cross-Validation N, input integer number bigger than 2

4.Classification method
* NaiveBayes
* Decision Tree
* Discriminant classification
* K-nearest neighbor(KNN)
* Suppoer vector mechine(SVM)
* Classification Ensemble
* Backpropagation Neural Network(BPN)
* Radial basis network(RBN)
* Adaptive neuro-fuzzy inference system(ANFIS)

Update
------------------------- 
v1.1
* Add 4 methods including Classification Ensemble, Backpropagation Neural Network(BPN)
  Radial basis network(RBN), and Adaptive neuro-fuzzy inference system(ANFIS)
* Show accuracy rate of classification

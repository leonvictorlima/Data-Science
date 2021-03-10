<h1 align="center">Credit Analysis using Machine Learning Classification Models</h1>

<h1 align="center">
  <img src="https://github.com/leonvictorlima/Machine-Learning-Credit-Analysis-Data-Science/blob/main/Images/credit.jpg"  width="500"/>
</h1>

<a name="introduction"></a>
## Introduction

Machine Learning  is one of the current technology and is being each time more relevant around the world. Its application is conducted and implemented in a large scale as well as managed a revolution.

This project is to share some knowledge about the use of machine learning in credit analysis. The implemantation of the model is straight to the point. By the way, like any machine learning application, the crucial steps are followed. Here they are:

Tabela de conteúdos
=================
<!--ts-->
   * [Introduction](#introduction)
   * [Business problem](#business-problem)
   * [Collecting the data](#collecting-data)
   * [Analysing data](#analysing-data)
   * [Feature Selection](#feature-selection)
   * [The choice of Machine Learning Model](#machine-learning)
   * [Training, Testing, and evaluate each of them](#training-test)
   * [Extra performance analysis](#perfomance-analysis)
   * [Conclusion](#conclusion)
   * [Bibliography](#bibliography)
<!--te-->

<a name="business-problem"></a>
**1) Business problem:**

The big idea here and behind all sector and also familiar for each financial institution is credit analysis. How prevent whether anyone will be a debtor or not? How to identify whether is a great oportunity to invest in anybody's dream? That are some relevant question to be answered.

<a name="collecting-data"></a>
**2) Collecting the data:**

The data is provided with elemental information which are used in credit analysis. The dataset is attached here.

```R
df<-read.csv("credit_dataset.csv")
```
<a name="analysing-data"></a>
**3) Analysing data:**

Normally it is a prominent part of all data science processes and some analyses are realized. On the other hand, the main aspect which is explored in this case study is related to the execution of some prediction prototype. For that reason, this point will not be diving deeply.

```R
# The idea here is not to use too much time in data analysis. Even though,
# being the crux in every work analysis. But the main point is distinct 
# ML models implementation.

# Fast analysis

head(df)
str(df)
summary(df)

# NA values?

sum(is.na(df))

### Plotting data was leaped ###

# Fucntion to convert variables to factor

tofactor <- function(dataset, variable){
  for (items in variable){
    dataset[[items]] <- as.factor(dataset[[items]])
  }
  return(dataset)
}

# Function to normalize the data

?scale

toscale <- function(dataset, variable){
  for (items in variable){
    dataset[[items]] <- scale(dataset[[items]], center = TRUE, scale = TRUE)
  }
  return(dataset)
}

# Setting parameters to make use of previous functions.

colnames(df)

factor_variables <- c("credit.rating","account.balance", "previous.credit.payment.status",
  "savings","credit.purpose","employment.duration","installment.rate","marital.status",
  "guarantor", "residence.duration", "current.assets", "other.credits",
  "apartment.type", "bank.credits","occupation","dependents", "telephone",
  "foreign.worker")

scale_variables <- c("credit.duration.months", "credit.amount", "age")

# Converting to factor and Normaling the data.

df <- tofactor(df,factor_variables)
df <- toscale(df,scale_variables)

# Taking a look at the data.

View(df)
str(df)
summary(df)

```
![](https://github.com/leonvictorlima/ML-Credit-Analysis-Data-Science/blob/main/Images/dataframe.JPG)

<a name="feature-selection"></a>
**4) Feature Selection:**

One of the most important moments is feature selection. Based on our dataset, the outcomes from any machine learning project and its capacity is explored at maximum level with an excellent feature selection. It needs a strong workforce to fit it.

```R
### 5) Perfoming feature selection.

# install.packages(randomForest)
library(randomForest)
?randomForest

feature_selection <- randomForest(credit.rating ~ ., data = train_data,
                                  ntrees = 100,
                                  nodesize = 1,
                                  importance=TRUE)

varImpPlot(feature_selection)

```
<img src="https://github.com/leonvictorlima/ML-Credit-Analysis-Data-Science/blob/main/Images/feature_selection.png" width="700">

```R
#We shall work with intertwine outcomes. We shall use all variables except two:
#dependents and telephone.
```
<a name="machine-learning"></a>
**5) The choice of Machine Learning Model:**

Our case study is a classification matter. Consequently, the choice to estimate and come up with a machine learning model will be related to this. For intensification and get multiple results, its approach is built by three separated methods: Naive Bayes, SVM (Support Vector Machine), and Random Forest models.

<a name="trainning-test"></a>
**6) Training, Testing, and evaluate each of them:**

It is the main segment of the project. The fundamental thing here is extract, testing, and understanding diverse options to analyse and interpret outcomes from distinct models applied. Particularly, the outcomes at this point are reflected from previous steps, and when necessary to change, optimize, refine outputs we must come back and readapt, modify, rearrange, and so on some antecedent stages. 

```R

### 7) Training, Test, and evaluate each of them.

# install.packages(e1071)
# install.packages(caTools)
# install.packages(caret)

library(e1071) 
library(caTools) 
library(caret) 

# Model 01 - Naive Bayes

?naiveBayes
naivebayes_model <- naiveBayes(credit.rating ~ .
                            - dependents
                            - telephone,
                            data = train_data)

?predict
nb_pred <- predict(naivebayes_model, newdata = test_data[,-1])

# Naive-Bayes - Confusion matrix

?confusionMatrix
confusionMatrix(table( real = test_data[,1], previsto = nb_pred))

```

![](https://github.com/leonvictorlima/ML-Credit-Analysis-Data-Science/blob/main/Images/NaiveBayes_cm.JPG)

```R

# Model 02 - SVM (Support Vector Machines)

?svm

svm_model = svm(formula = credit.rating ~ .
                 - dependents
                 - telephone, 
                 data = train_data, 
                 type = 'C-classification')
svm_model

svm_pred <- predict(svm_model, newdata = test_data[,-1])

# SVM - Confusion matrix

confusionMatrix(table(real = test_data[,1], predicted = svm_pred))

```
![](https://github.com/leonvictorlima/ML-Credit-Analysis-Data-Science/blob/main/Images/svm_cm.JPG)

```R

# Model 03 - Random Forest

?randomForest

rf_model <- randomForest(credit.rating ~ .
             - dependents
             - telephone,
             data = train_data,
             ntrees = 100,
             nodesize = 1)

rf_model
rf_pred <- predict(rf_model, newdata = test_data[,-1])

# Random Forest - Confusion matrix

confusionMatrix(table(real = test_data[,1],predicted = rf_pred))

```
![](https://github.com/leonvictorlima/ML-Credit-Analysis-Data-Science/blob/main/Images/RF_cm.JPG)

<a name="permonance-analysis"></a>
**7) Extra performance analysis:**

The last and additional context here is one of most important evaluation metrics which is checking Area Under Curve (AUC) + Receiver Operating Characteristics (ROC). In this case, these element are used to complement and reinforce the confidence estimated previously. 

```R
####### Extra Study ########

# Below is other way to check model perfomance employing AUC-ROC Curve
# for each one prototype applied previously.

### Area Under the Curve - ROC curve ##

# The AUC (Area Under the Curve) - ROC (Receiver Operating Characteristics)
# Curve is one important evaluation metric for classification model perfomance.
# In order to implement and explore that, let's define its attributes here.

# install.packages("ROCR")
library("ROCR")

## ROC

###### Fuction to Plot AUC-ROC curve ######

par(mfrow = c(1,2))

plot.roc.curve <- function(predictions, title.text){
  perf <- performance(predictions, "tpr", "fpr")
  plot(perf, col = "black", lty = 1, lwd = 2, 
       main = title.text, cex.main = 0.6, 
       cex.lab = 0.8, xaxs="i", yaxs="i")
  abline(0,1, col = "red")
  auc <- performance(predictions, "auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,2)
  legend(0.4,0.4, legend = c(paste0("AUC: ",auc)), cex = 0.6, bty = "n", box.col = "white")
  
}

plot.pr.curve <- function(predictions, title.text){
  perf <- performance(predictions, "prec", "rec")
  plot(perf, col = "black", lty = 1, lwd = 2,
       main = "title.text", cex.main = 0.6, cex.lab = 0.8, xaxs = "i", yaxs = "i")
}

#################

#### Model 01 - Naive-Bayes - AUC-ROC curve Analysis.

# Producing a matrix with real and predicted data to generate ROC Curve.

matrix_nb <- as.data.frame(cbind(nb_pred, test_data[,1]))
colnames(matrix_nb) <- c("predicted", "real")

# Naive-Bayes - ROC Curve

pred_nb <- prediction(matrix_nb$predicted, matrix_nb$real)

#par(mfrow = c(1,2))

plot.roc.curve(pred_nb, title.text = "Naive Bayes - ROC Curve")
plot.pr.curve(pred_nb, title.text = "Precision/Recall Curve ")

```
![](https://github.com/leonvictorlima/ML-Credit-Analysis-Data-Science/blob/main/Images/NaiveBayes_roc.png)

```R

### Model 02 - SVM - AUC-ROC curve Analysis.

# Producing a matrix with real and predicted data to generate ROC Curve.

matrix_svm <- as.data.frame(cbind(svm_pred, test_data[,1]))
colnames(matrix_svm) <- c("predicted", "real")

# SVM - ROC Curve

pred_svm <- prediction(matrix_svm$predicted, matrix_svm$real)

#par(mfrow = c(1,2))

plot.roc.curve(pred_svm, title.text = "SVM - Curva ROC")
plot.pr.curve(pred_svm, title.text = "Curva Precision/Recall")

```

![](https://github.com/leonvictorlima/ML-Credit-Analysis-Data-Science/blob/main/Images/svm_roc.png)

```R
#### Model 03 - Random Forest - AUC-ROC curve Analysis.

# Producing a matrix with real and predicted data to generate ROC Curve.

matrix_rf <- as.data.frame(cbind(rf_pred, test_data[,1]))
colnames(matrix_rf) <- c("predicted", "real")

# Random Forest - ROC Curve

pred_rf <- prediction(matrix_rf$predicted, matrix_rf$real)

par(mfrow = c(1,2))

plot.roc.curve(pred_rf, title.text = "RF - Curva ROC")
plot.pr.curve(pred_rf, title.text = "RF - Curva Precision/Recall")
```
![](https://github.com/leonvictorlima/ML-Credit-Analysis-Data-Science/blob/main/Images/rf_roc.png)

<a name="conclusion"></a>
## Conclusion

At the end of the first investigation the model 03, Random Forest, had a better performance than others. On the other hand, we must consider that it was a quick exploration and any aspects were not contemplated just now. When making use of AUC-ROC Curve is easy to realize its incisive achievement. Further that, it is important to return and intensify exploration to get elevate more its efficiency. 

As a suggestion for future works, it is necessary to come up with, or rearrange, new data modulation in the manipulation step, as well as replace, take out, and make use of distinct variables in the feature selection step. In addition, conduct a new version of machine learning design performed manipulating its arguments.

<a name="bibliography"></a>
## Bibliography

[1] dataset - https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

[2] Data Science Academy - https://www.datascienceacademy.com.br/

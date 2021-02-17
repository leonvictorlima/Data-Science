## The use of machine learning models for analysis is one of the most 
# provinient and usual in the data science area.

# The ability to predict and boosting the time analysis generates excellent
# response on evaluate and classification of debtors as well as contributes
# to improve accuracy in credit loans as well. 

# This project is a simple example of the use of machine learning models
# with elementary credit dataset to data scientist beginners.

# This scenario of study is divided into some steps: 

# 1) Get dataset.*
# 2) Load dataset.
# 3) Studying and Manipulate/Transform dataset.
# 4) Splitting data into Test and Training sets.
# 5) Perfoming feature selection.
# 6) Define and choosing different ML models.
# 7) Training, Test, and evaluate each of them. 

#############

# define your work directory.

getwd()
setwd("C:/FCD/BigDataRAzure/Cap18/")

### 1) Get dataset.

# * The dataset is provided to use here or others study case.

### 2) Load dataset.

df<-read.csv("credit_dataset.csv")

### 3) Studying and Manipulate/Transform dataset.

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

### 4) Splitting data into Test and Training sets.

# Train = 70%
# Test = 30%

?sample
train_size <- floor(0.7 * nrow(df))
train_sample <- sample(c(1:nrow(df)),train_size, replace = FALSE)

train_data <- df[train_sample,]
test_data <- df[-train_sample,]


### 5) Perfoming feature selection.


# install.packages(randomForest)
library(randomForest)
?randomForest

feature_selection <- randomForest(credit.rating ~ ., data = train_data,
                                  ntrees = 100,
                                  nodesize = 1,
                                  importance=TRUE)

varImpPlot(feature_selection)

#Work with intertwine outcomes. We shall use all variables except two:
#dependents and telephone.

### 6) Define and choosing different ML models.

# Selecting model afterward Feature-Selection. Our case study is a
# classification matter. For that reason, the choice to estimate 
# circumstance are three different models: Naive-Bayes, SVM, and 
# Random Forest.

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


### Model 02 - SVM - AUC-ROC curve Analysis.

# Producing a matrix with real and predicted data to generate ROC Curve.

matrix_svm <- as.data.frame(cbind(svm_pred, test_data[,1]))
colnames(matrix_svm) <- c("predicted", "real")

# SVM - ROC Curve

pred_svm <- prediction(matrix_svm$predicted, matrix_svm$real)

#par(mfrow = c(1,2))

plot.roc.curve(pred_svm, title.text = "SVM - Curva ROC")
plot.pr.curve(pred_svm, title.text = "Curva Precision/Recall")

#### Model 03 - Random Forest - AUC-ROC curve Analysis.

# Producing a matrix with real and predicted data to generate ROC Curve.

matrix_rf <- as.data.frame(cbind(rf_pred, test_data[,1]))
colnames(matrix_rf) <- c("predicted", "real")

# Random Forest - ROC Curve

pred_rf <- prediction(matrix_rf$predicted, matrix_rf$real)

par(mfrow = c(1,2))

plot.roc.curve(pred_rf, title.text = "RF - Curva ROC")
plot.pr.curve(pred_rf, title.text = "RF - Curva Precision/Recall")

?plot.pr
?plot.roc.curve


# Conclusion

# In the end of first investigation the model 01, Naive Bayes, has a 
# better performance than others. On the other hand, we must consider
# that it was a quick exploration and any aspects were not contemplated
# just now. 

# As suggestion to future works, it is necessary come up with, or rearrange,
# new data modelation in manipulation step, as well as replace, take off, and
# make use of distinct variables in feature selection step. In addition, 
# conduct a new version of machine learning design performed manipulating
# its arguments.




"""
Install all the libraries if running for first time by uncommenting all the install scripts
"""

#install.packages("tidyr",repos = "http://cran.us.r-project.org")
#install.packages("ggplot2",repos = "http://cran.us.r-project.org")
#install.packages("ggpubr",repos = "http://cran.us.r-project.org")
#install.packages("NISTunits", dependencies = TRUE,repos = "http://cran.us.r-project.org")
#install.packages("corrplot",repos = "http://cran.us.r-project.org")
#install.packages("Hmisc",repos = "http://cran.us.r-project.org")
#install.packages("dplyr",repos = "http://cran.us.r-project.org")
#install.packages("ISLR",repos = "http://cran.us.r-project.org")
#install.packages("rpart",repos = "http://cran.us.r-project.org")
#install.packages("randomForest",repos = "http://cran.us.r-project.org")
#install.packages("lubridate",repos = "http://cran.us.r-project.org")
#install.packages("caret",repos = "http://cran.us.r-project.org")
#install.packages("mlbench",repos = "http://cran.us.r-project.org")
#install.packages("tidyverse",repos = "http://cran.us.r-project.org")
#install.packages("car",repos = "http://cran.us.r-project.org")
#install.packages("regclass",repos = "http://cran.us.r-project.org")
#install.packages("usdm",repos = "http://cran.us.r-project.org")
#install.packages("fmsb",repos = "http://cran.us.r-project.org")
#install.packages("dummies",repos = "http://cran.us.r-project.org")
#install.packages('Boruta',repos = "http://cran.us.r-project.org")
#install.packages('ROCR',repos = "http://cran.us.r-project.org")
#install.packages("RDocumentation",repos = "http://cran.us.r-project.org")
#install.packages("pROC",repos = "http://cran.us.r-project.org")
rm(list = ls())

##Load Libraries
library(pROC)
library(ROCR)
library(RDocumentation)
library(lubridate)
library(rpart)
library(randomForest)
library(mlbench)
library(caret)
library("ISLR")
library("Hmisc")
library("dplyr")
library("tidyr")
library("ggplot2")
library(NISTunits)
library(corrplot)
library(fmsb)
library(dummies)
library(Boruta)
set.seed(0)

#############################INTRODUCTION TO PROBLEM STATEMENT#################################

"""
Santander Customer Transaction Prediction
    Background -
At Santander, mission is to help people and businesses prosper. We are always looking
for ways to help our customers understand their financial health and identify which
products and services might help them achieve their monetary goals.
Our data science team is continually challenging our machine learning algorithms,
working with the global data science community to make sure we can more accurately
identify new ways to solve our most common challenge, binary classification problems
such as: is a customer satisfied? Will a customer buy this product? Can a customer pay
this loan?
  Problem Statement -
In this challenge, we need to identify which customers will make a  transaction in
the future, irrespective of the amount of money transacted.
  You are provided with an anonymized dataset containing numeric feature variables, the
binary target column, and a string ID_code column. The task is to predict the value
of target column in the test set.
As from the problem statement it is confirmed that the problem that we are going to solve is binary classification problem</h4>
In this problem we have to predict target variable which is 0 or 1?
"""
##############################################################################################

#changing directory for working
setwd("/home/sushant/machine_learning_cab_project/santander_customer_transaction/")

### Read the data
df = read.csv("train.csv", header = T, as.is = T)
df_test = read.csv("test.csv", header = T, as.is = T)

############################DATA EXPLORING####################################################
#checking for dimensions
print(dim(df))
#we have 200000 rows and 200 columns in  total
print(summary(df))
#As we can see there are no missing values in a dataframe so we can move further

#Now we will do outlier analysis as outlier can be one noisy and can make our model corrupted
boxplot(df$var_0)
boxplot(df$var_1)
boxplot(df$var_10)

#If we observe boxplot properly we can see there are outliers some points are below the whiskers 
#as well as some points we have are beyond upper whiskers but outliers here doesn't seems to be noise as 
#there value are close enough them maximum and minimum so we will not remove it

#let us see the count of class labels 0 and 1
counts <- table(df$target)
barplot(counts, main="Class distribution",
        xlab="Number of class")
print(counts)
#From the above barplot we can see there are large number rows with class 0 and very less number
#of rows of class 1,hence from this we can conclude that we are having very imbalanced dataset.
##########################################################################################

#################################HISTOGRAM PLOTS##########################################
#Let us check the distribution of some variable to get the insights about the vaiable
hist(df$var_0)
hist(df$var_1)
hist(df$var_10)
#From the above histograms we see mostly all the variables are having same distributions
##########################################################################################

####################################PAIR PLOTS############################################
#Let us check th pairwise class distribution of some variables
pairs(df[3:6], 
      main = "Pairplots",
      pch = 21, 
      bg = c("#1b9e77", "#d95f02", "#7570b3")[unclass(df$target)])
##########################################################################################

#####################################CORRELATION PLOTS####################################
#Now let us check the correlation between variables
df_for_corr <- select(df,-c(ID_code,target))
M<-cor(df_for_corr)
head(round(M,2))
corrplot(M, method="color")
#As from the correlation plot we can see that there is no correlation between variables
##########################################################################################

####################################MULTICOLLINEARITY DETECTION##########################
#Let us check for the vif factor which will help us to detect multicollinearity
for(i in names(df_for_corr)){
 print(i)
 VIF(lm(i ~ ., data = df_for_corr))
}
##CHECKING FOR MULTICOLLINEARITY
#I HAVE DONE MULTICOLLINEARITY FOR ALL THE VARIABLES BUT FOR SAMPLE WE WILL ONLY USE 3 
VIF(lm(var_0 ~ ., data = df_for_corr))
VIF(lm(var_1 ~ ., data = df_for_corr))
VIF(lm(var_10 ~ ., data = df_for_corr))
#############################################################################################

#######################################SAMPLING DATA FOR SPEEDING UP#########################
#df_without_code <- select(df,-c(ID_code))
#As this is very difficult for us to operaate on this size of data so we will 
#sample the data 1000 and operate on that
df_sample <- df[sample(nrow(df), 1000), ]
counts <- table(df_sample$target)
print(counts)
#############################################################################################

####################################FEATURE ENGINEERING######################################
# removing index from sample
df_sample_without_index <- select(df_sample,-c(ID_code))
# Perform Boruta search for feature importance
boruta_output <- Boruta(target ~ ., data=na.omit(df_sample_without_index), doTrace=0)  
#printing name of most important features
# Get significant variables including tentatives
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  
## split data into 80:20
smp_size <- floor(0.80 * nrow(df_sample_without_index))
## set the seed to make your partition reproducible
set.seed(100)
train_ind <- sample(seq_len(nrow(df_sample_without_index)), size = smp_size)
x_train <-df_sample_without_index[train_ind, ]
x_test <- df_sample_without_index[-train_ind, ]
###############################################################################################

###########################TRYING TEST AND TRAIN DATA ON DIFFERENT MODELS######################

#####################################LOGISTIC REGRESSION#######################################
fit <- glm(target ~., data = x_train, family = binomial, maxit = 100)
summary(fit)
predict <- predict(fit,newdata=x_train,type="response")
predict_train <- ifelse(predict > 0.5,1,0)
confusionMatrix = table(x_train$target,predict_train)
print(confusionMatrix)

#let us try this on test data  and print roc_auc score and roc curve
predict <- predict(fit,newdata=x_test,type="response")
predict_test <- ifelse(predict > 0.5,1,0)
confusionMatrix = table(x_test$target,predict_test)
print(confusionMatrix)
pred <- prediction(predict_test, x_test$target)
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)
print((x_test$target))
print((predict_test))
roc_obj <- roc(x_test$target,predict_test)
auc(roc_obj)
#False Negative rate
#FNR = FN/FN+TP 
#precsion =  TP/TP+FP
############################################################################################

########################################DECSION TREES#######################################
fit <- rpart(target ~ ., data=x_train)
predict <- predict(fit,newdata=x_train, method="response")
predict_train <- ifelse(predict > 0.5,1,0)
print(predict_train)
print(dim(x_train$target))
confusionMatrix = table(x_train$target,predict_train)
print(confusionMatrix)
#let us try this on test data
predict <- predict(fit,newdata=x_test, method="response")
predict_test <- ifelse(predict > 0.5,1,0)
confusionMatrix = table(x_test$target,predict_test)
print(confusionMatrix)
pred <- prediction(predict_test, x_test$target)
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)
roc_obj <- roc(x_test$target, predict_test)
auc(roc_obj)
#False Negative rate
#FNR = FN/FN+TP 
#precsion =  TP/TP+FP
#############################################################################################

#########################################RANDOM FORESTS######################################
rf <- randomForest(
  target ~ .,
  data=x_train
)
predict <- predict(rf,newdata=x_train,type="response")
predict_train <- ifelse(predict > 0.5,1,0)
print(predict_train)
print(dim(x_train$target))
confusionMatrix = table(x_train$target,predict_train)
print(confusionMatrix)
#let us try this on test data
predict <- predict(rf,newdata=x_test,type="response")
predict_test <- ifelse(predict > 0.5,1,0)
confusionMatrix = table(x_test$target,predict_test)
print(confusionMatrix)
pred <- prediction(predict_test, x_test$target)
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)
roc_obj <- roc(x_test$target, predict_test)
auc(roc_obj)
#False Negative rate
#FNR = FN/FN+TP 
#precsion =  TP/TP+FP
################################################################################################


####################################TAKE AWAY FROM DIFFERNT MODELS####################################
#1.From all the models that we have logistic regression was having better auc_roc score
#2.From all the models that we have tried logistic regression was having better precision and recall
#3.From all the models that we have tried logistic regression was having better speed so we would accept logistic 
#regression for use on production enviroment
#######################################################################################################


####################################PREDICTING REAL TEST DATA#####################################
### Read the test
df_test = read.csv("test.csv", header = T, as.is = T)
print(dim(df_test))
# removing index from sample
df_test <- select(df_test,-c(ID_code))
predict <- predict(fit,newdata=df_test,type="response")
df_test$predict <- ifelse(predict > 0.5,1,0)
##################################################################################################







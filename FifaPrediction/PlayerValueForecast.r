library(MASS)
library(tidyverse)
library(tree)
library(PerformanceAnalytics)
library(corrplot)

options(scipen=999)

Players<-read.csv2("complete.csv",stringsAsFactors = F, sep=",",dec=".")
names(Players)

# a ) First drop all goalkeepers from the dataset. 

dataset <- Players[is.na(Players$gk), ]
dataset <- dataset[1:2000,]

# b ) Analyze your predictors/independent variables using techniques shown in the class.

summary(dataset)

correlations <- cor(dataset[,c (3,12,17,20,21)])
corrplot(correlations, method="circle")

pairs (dataset[,c(3,12,17,20,21)])
chart.Correlation(dataset[,c(3,12,17,20,21)], histogram=TRUE, pch=19)

# c ) Take the first 1000 observation as the training set and use the rest as the test set.

dataset <- dataset[,3:37]

dataset$work_rate_att <- as.factor(dataset$work_rate_att)
dataset$work_rate_def <- as.factor(dataset$work_rate_def)
dataset$preferred_foot <- as.factor(dataset$preferred_foot)

train <- dataset[1:1000,]
test <- dataset[1001:nrow(dataset),]
str(dataset)
names(dataset)



# d ) Fit a multiple regression model to predict �eur_value�. Use only the training set to fit the regression model.

lm.fit <- lm(formula = eur_value ~.  , data = train)
confint(lm.fit)
coef(lm.fit)

# e) Analyze your estimated regression models. Comment on coefficients, adjusted R square and F statistic of the model. 

summary(lm.fit)

# f) Predict �eur_value� in the test set using the regression model obtained in (d).

testpredict<-predict(lm.fit, newdata=test)

#MSE

MSE<-(1/nrow(test))*sum((test$eur_value-testpredict)^2)
RMSE <- sqrt(MSE)
RMSE1 <- log(RMSE)

#g) Fit a Ridge model and a Lasso model to predict �eur_value�. Use only the training set to
#fit these regression models. Determine the lambda parameter using cross-validation.

library(glmnet)

#Ridge
set.seed(1)
train.mat <- model.matrix(eur_value ~ ., data = train)
test.mat <- model.matrix(eur_value ~ ., data = test)

grid <- 10 ^ seq(10, -2, length = 100)
fit.ridge <- glmnet(train.mat, train$eur_value, alpha = 0, lambda = grid)
cv.ridge <- cv.glmnet(train.mat, train$eur_value, alpha = 0, lambda = grid)
bestlam.ridge <- cv.ridge$lambda.min
bestlam.ridge
plot(cv.ridge)
log(bestlam.ridge)

#Lasso
set.seed(1)
grid <- 10 ^ seq(10, -2, length = 100)
fit.lasso <- glmnet(train.mat, train$eur_value, alpha = 1, lambda = grid)
cv.lasso <- cv.glmnet(train.mat, train$eur_value, alpha = 1, lambda = grid)
bestlam.lasso <- cv.lasso$lambda.min
bestlam.lasso
plot(cv.lasso)
log(bestlam.lasso)

#h) Analyze your Lasso Model. Compare your Lasso Model with the multiple regression model estimated in (d).

predict(fit.lasso, s = bestlam.lasso, type = "coefficients")
plot(cv.lasso$glmnet.fit, "lambda", label=TRUE)

#i) Predict �eur_value� in the test set using the Ridge model and the Lasso model obtained in (g). 
#Calculate MSEs of these models only using the test set.

pred.ridge <- predict(fit.ridge, s = bestlam.ridge, newx = test.mat)
MSE2 <- mean((pred.ridge - test$eur_value)^2)
RMSE2 <- log(sqrt(MSE2))

pred.lasso <- predict(fit.lasso, s = bestlam.lasso, newx = test.mat)
MSE3 <- mean((pred.lasso - test$eur_value)^2)
RMSE3 <- log(sqrt(MSE3))

#j) Fit a regression tree to predict �eur_value�. Use only the training set to fit the regression 
# model. Determine the number of terminal nodes using cross-validation.

library(tree)
set.seed(1)
tree.players <- tree(eur_value~., data = train)
summary(tree.players)
plot(tree.players)
text(tree.players)

cv.players <- cv.tree(tree.players)
plot(cv.players$size,cv.players$dev,type='b')

prune.players <- prune.tree(tree.players,best=9)
plot(prune.players)
text(prune.players,pretty=0)

# k) Predict �eur_value� in the test set using the regression tree model obtained in (j).
#Calculate the MSEs of the regression tree only using the test set. 

yhat <- predict(prune.players,newdata=test)
boston.test <- test[,"eur_value"]
MSE4 <- mean((yhat-boston.test)^2)
RMSE4 <- log(sqrt(MSE4))

#l) Fit random forests to predict �eur_value�. Use only the training set to fit the regression
#model. Determine the number of variables used in each split using the cross-validation.
#Grow 500 trees for random forest.
library(randomForest)
set.seed(1)

cvplayer <- train[sample(nrow(train)),]
folds <- cut(seq(1,nrow(cvplayer)),breaks=5,labels=FALSE)

total_mse <- rep(NA,34)
for (i in 1:34) {
  mse <- rep(NA,5)
  #5-fold cross validation
  for (t in 1:5){
    set.seed(1)
    cv_test_index <- which(folds==t,arr.ind=TRUE)
    cv_train <- train[-cv_test_index,]
    cv_test <- train[cv_test_index,]
    rf.players <- randomForest(eur_value~., data=cv_train, mtry= i, 
                               ntree=500, importance=TRUE, na.action = na.omit)
    pred <- predict(rf.players,newdata=cv_test)
    mse[t] <- (1/nrow(cv_test))*sum((pred-cv_test$eur_value)^2)
  }
  total_mse[i] <- mean(mse)
}

min_mtry <- which.min(total_mse)
min_mtry

# m) According to random forests, which variables are import? Comment.

rf.players <- randomForest(eur_value~., data=train, mtry=min_mtry, 
                           ntree=500, importance=TRUE, na.action = na.omit)
importance(rf.players, type=1)
varImpPlot(rf.players, type=1)

#n) Predict �eur_value� in the test set using the random forest model obtained in (l).
#Calculate the MSEs of the random forest only using the test set.

set.seed(1)
rf.players <- randomForest(eur_value~., data=train, mtry=min_mtry, 
                           ntree=500, importance=TRUE, na.action = na.omit)
yhat.rf <- predict(rf.players,newdata=test)

MSE5 <- mean((yhat.rf-test$eur_value)^2)
RMSE5 <- log(sqrt(MSE5))

#o) Compare MSEs obtained in (f), (i), (k) and (n)
c(RMSE1,RMSE2,RMSE3,RMSE4,RMSE5)
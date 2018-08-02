IonoTrain=read.csv("C:/Users/Jeffry/Desktop/TAMU Research/Data Analytics/Final Project/Train.csv")
IonoTest=read.csv("C:/Users/Jeffry/Desktop/TAMU Research/Data Analytics/Final Project/Test.csv")
ionosphere=read.csv("C:/Users/Jeffry/Desktop/TAMU Research/Data Analytics/Final Project/Data.csv")
IonoTrain=IonoTrain[,-2]
IonoTest=IonoTest[,-2]
ionosphere=ionosphere[,-2]
# Logistic Regression
logistic_model = glm(V35~. -V1-V10-V33-V25-V7-V28-V20-V23-V12-V24-V17-V19-V29-V13-V21-V15-V11-V34-V30-V14-V18-V32-V6-V4-V16-V9-V8-V31, data = IonoTrain, family = binomial)
summary(logistic_model)
logistic_model1=glm(V35~V3+V5+V22+V26+V27, data=IonoTrain,family=binomial)
summary(logistic_model1)

logistic_probs=predict(logistic_model1,IonoTest,type="response")
logistic_pred_y = rep("b", length(IonoTest$V35))
logistic_pred_y[logistic_probs > 0.5] = "g"
table(logistic_pred_y,IonoTest$V35)
mean(logistic_pred_y==IonoTest$V35)
mean(logistic_pred_y!=IonoTest$V35)

#CV for Glm

library(boot)
cv_error=cv.glm(IonoTest,logistic_model1,K=5)$delta[1]
cv_error

#ROC for Glm

library(ROCR)
library(pROC)
set.seed(5)
rocplot=function(pred,truth,...)
{
  predob=prediction(pred,truth)
  perf=performance(predob,"tpr","fpr")
  plot(perf,...)
}
rocplot(pred=logistic_probs,truth=IonoTest$V35)
abline(a=0,b=1)
auc.tmp = performance(prediction(logistic_probs,IonoTest$V35),"auc");
auc.glm = as.numeric(auc.tmp@y.values);
auc.glm 

# Linear Discriminant Analysis (LDA)
library(MASS)
lda.model = lda(V35~V3+V5+V22+I(V26^2)+V27, data=IonoTrain)
lda_pred_y = predict(lda.model, IonoTest)
table(lda_pred_y$class,IonoTest$V35)
mean(lda_pred_y$class == IonoTest$V35)
mean(lda_pred_y$class != IonoTest$V35)

#CV for Lda
library(sortinghat)
X=cbind(IonoTrain$V3,IonoTrain$V5,IonoTrain$V22,IonoTrain$V26,IonoTrain$V27)
Y=IonoTrain$V35
lda_wpa=function(object,newdata){predict(object,newdata)$class}
errorest_cv(x=X,y=Y,train = lda,classify = lda_wpa,num_folds = 10)

#ROC for Lda
rocplot=function(pred,truth,...)
{
  predob=prediction(pred,truth)
  perf=performance(predob,"tpr","fpr")
  plot(perf,...)
}

rocplot(pred =lda_pred_y$posterior[,2],truth=IonoTest$V35);
abline(a=0,b=1)
auc.tmp = performance(prediction(lda_pred_y$posterior[,2],IonoTest$V35),"auc");
auc.lda = as.numeric(auc.tmp@y.values);
auc.lda 

# Quadratic Discriminant Analysis (QDA)

qda.model = qda(V35~V3+V5+V22+I(V26^2)+V27, data=IonoTrain)
qda_pred_y = predict(qda.model, IonoTest)
table(qda_pred_y$class,IonoTest$V35)
mean(qda_pred_y$class == IonoTest$V35)
mean(qda_pred_y$class != IonoTest$V35)
# CV for Qda
qda_wpa=function(object,newdata){predict(object,newdata)$class}
errorest_cv(x=X,y=Y,train = qda,classify = qda_wpa,num_folds = 10)

#ROC for Qda
set.seed(5)
rocplot=function(pred,truth,...)
{
  predob=prediction(pred,truth)
  perf=performance(predob,"tpr","fpr")
  plot(perf,...)
}

rocplot(pred =qda_pred_y$posterior[,2],truth=IonoTest$V35);
abline(a=0,b=1)
auc.tmp = performance(prediction(qda_pred_y$posterior[,2],IonoTest$V35),"auc");
auc.qda = as.numeric(auc.tmp@y.values);
auc.qda

#KNN 
library (class)
set.seed (1)
train.X=cbind(IonoTrain$V3,IonoTrain$V5,IonoTrain$V22,IonoTrain$V26,IonoTrain$V27)
test.X=cbind(IonoTest$V3,IonoTest$V5,IonoTest$V22,IonoTest$V26,IonoTest$V27)
train.Y=cbind(IonoTrain$V35)
test.Y=cbind(IonoTest$V35)
knn.pred=knn(train.X,test.X,train.Y,k=3)
table(knn.pred,test.Y)
mean(knn.pred==test.Y)
mean(knn.pred!=test.Y)

#CV for KNN
x=cbind(IonoTrain$V3,IonoTrain$V5,IonoTrain$V22,IonoTrain$V26,IonoTrain$V27)
y=IonoTrain$V35
out= knn.cv(x,y,k=5)
Error = 1 - sum(abs(y == out)) / length(out)
Error

#ROC for KNN
set.seed(5)
rocplot=function(pred,truth,...)
{
  predob=prediction(pred,truth)
  perf=performance(predob,"tpr","fpr")
  plot(perf,...)
}
set.seed(6)
knn.pred=knn(data.frame(train.X),data.frame(test.X),IonoTrain$V35,k=32,prob = TRUE)
rocplot(pred=attr(knn.pred,"prob"),truth = IonoTest$V35)
abline(a=0,b=1)
auc.tmp = performance(prediction(attr(knn.pred,"prob"),IonoTest$V35),"auc");
auc.knn = as.numeric(auc.tmp@y.values);
auc.knn 

# Simple Classification Trees
library (ISLR)
library (tree)
attach(IonoTrain)
tree.Data =tree(V35~V3+V5+V22+I(V26^2)+V27,IonoTrain)
summary (tree.Data)
plot(tree.Data)
text(tree.Data,pretty =0)

tree.pred=predict(tree.Data,IonoTest,type="class")
table(tree.pred,IonoTest$V35)
mean(tree.pred==IonoTest$V35)
mean(tree.pred!=IonoTest$V35)

set.seed (3)
cv.Data =cv.tree(tree.Data,FUN=prune.misclass )
cv.Data
cv.Data$size
cv.Data$dev
plot(cv.Data)

par(mfrow =c(1,2))
plot(cv.Data$size ,cv.Data$dev ,type="b")
plot(cv.Data$k ,cv.Data$dev ,type="b")

# Pruning. 
prune.data = prune.misclass(tree.Data ,best =3)
plot(prune.data)
text(prune.data,pretty =0)
tree.pred1=predict(prune.data,IonoTest,type="class")
table(tree.pred1,IonoTest$V35)
mean(tree.pred1==IonoTest$V35)
mean(tree.pred1!=IonoTest$V35)



#Bagging. 
library(randomForest)
bag.model=randomForest(V35~V3+V5+V22+V26+V27,data=IonoTrain, mtry=5,importance=TRUE)
summary(bag.model)

bag.pred = predict (bag.model ,IonoTest)
plot(bag.pred ,IonoTest$V35)
abline (0,1)
table(bag.pred ,IonoTest$V35)
mean((bag.pred==IonoTest$V35)^2)
mean((bag.pred!=IonoTest$V35)^2)



# Random Forest.
library(caret)
set.seed(56)
rf.model=randomForest(V35~V3+V5+V22+V26+V27,data=IonoTrain, mtry=3,importance=TRUE)
summary(rf.model)
rf.pred = predict (rf.model ,IonoTest)
plot(rf.pred ,IonoTest$V35)
abline (0,1)
table(rf.pred ,IonoTest$V35)
mean((rf.pred==IonoTest$V35)^2)
mean((rf.pred!=IonoTest$V35)^2)

kfold5=createFolds(y = IonoTrain$V35,k = 5,list = TRUE)
randomforest.errorrate=rep(0,5)
for(j in 1:5)
{
  random_forest=randomForest(V35~V3+V5+V22+V26+V27,data=IonoTrain[-kfold5[[j]],],ntree=500, mtry=3, importance=TRUE)
  random_forest.prediction=predict(random_forest,IonoTest[kfold5[[j]],])
  randomforest.errorrate[j]=mean(random_forest.prediction!=IonoTest[kfold5[[j]],"V35"])
}
mean(randomforest.errorrate)



#ROC for tree

tree.predict=predict(prune.data,IonoTest,type="vector")
rocplot=function(pred,truth,...)
{
  predob=prediction(pred,truth)
  perf=performance(predob,"tpr","fpr")
  plot(perf,...)
}
length(IonoTest$V35)
rocplot(pred=tree.predict[,2],truth=IonoTest$V35);
abline(a=0,b=1)
auc_tmp = performance(prediction(tree.predict[,2],IonoTest$V35),"auc");
auc.rf = as.numeric(auc_tmp@y.values);
auc.rf
#SVC
set.seed(1)
install.packages("e1071")
library(e1071)
tune.out=tune(svm,V35~V3+V5+V22+I(V26^2)+V27,data=IonoTrain ,kernel="linear",ranges=list(cost=c(0.001, 0.01,0.1, 1,5,10)))
#SVM
tune.out=tune(svm, Response~V3+V5+V22+I(V26^2)+V27, data=IonoTrain, kernel="radial",ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
#ROC Curve for svm
install.packages("ROCR")
library(ROCR)


rocplot=function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}
svmfit.opt=svm(V35~V3+V5+V22+V26+V27, data=IonoTrain, kernel="radial", gamma= 0.5, cost=1,decision.values=T)
fitted=(predict(svmfit.opt,IonoTrain,decision.values=FALSE))
par(mfrow=c(1,2))
rocplot(as.numeric(fitted),ionosphere[IonoTrain,"V35"],main="IonoTraining Data")
svmfit.flex=svm(V35~V3+V5+V22+V26+V27, data=IonoTrain, kernel="radial", gamma= 50, cost=1, decision.values=T)
fitted=predict(svmfit.flex,IonoTrain,decision.values=F)
rocplot(as.numeric(fitted),ionosphere[IonoTrain,"V35"],add=T,col="red")
fitted= predict(svmfit.opt,IonoTest,decision.values=F)
rocplot(as.numeric(fitted),IonoTest$V35,main="Test Data")
fitted= predict(svmfit.flex,IonoTest,decision.values=F)
rocplot(as.numeric(fitted),IonoTest$V35,add=T,col="red")

area.svm= performance(prediction(as.numeric(fitted),IonoTest$V35),"auc")
as.numeric(area.svm@y.values)

#Neural Networks
install.packages("neuralnet")
require(neuralnet)
response=ifelse(IonoTrain$V35=="g",1,0)
nn=neuralnet(V35~V3+V5+V22+I(V26^2)+V27,data=IonoTrain,hidden=2,err.fct="ce",linear.output=FALSE)
nn
plot(nn)
nn$net.result
nn$weights
nn$result.matrix
nn$covariate
nn$net.result[[1]]
nn1=ifelse(nn$net.result[[1]]>0.5,1,0)
miscclassificationError=mean(response!=nn1)
miscclassificationError
OutputvsPred=cbind(response,nn1)
OutputvsPred
ci=confidence.interval(nn,alpha=0.05)
ci
par(mfrow=c(2,2))
gwplot(nn,selected.covariate="V3",min=-1,max=1)
gwplot(nn,selected.covariate="V5",min=-1,max=1)
gwplot(nn,selected.covariate="V22",min=-1,max=1)
gwplot(nn,selected.covariate="I(V26^2)",min=-1,max=1)
gwplot(nn,selected.covariate="V27",min=-1,max=1)


# New Model

# Random Forest Feature Selection
RF=randomForest(V35~.,data=IonoTrain, importance=T)
varImpPlot(RF)

# Logistic Regression

logistic_model2=glm(V35~V3+V5+V7+V8+V27, data=IonoTrain,family=binomial)
summary(logistic_model2)

logistic_probs2=predict(logistic_model2,IonoTest,type="response")
logistic_pred_y2= rep("b", length(IonoTest$V35))
logistic_pred_y2[logistic_probs2 > 0.5] = "g"
table(logistic_pred_y2,IonoTest$V35)
mean(logistic_pred_y2==IonoTest$V35)
mean(logistic_pred_y2!=IonoTest$V35)

#CV for Glm

library(boot)
cv_error=cv.glm(IonoTest,logistic_model2,K=5)$delta[1]
cv_error

#ROC for Glm

library(ROCR)
library(pROC)
set.seed(5)
rocplot=function(pred,truth,...)
{
  predob=prediction(pred,truth)
  perf=performance(predob,"tpr","fpr")
  plot(perf,...)
}
rocplot(pred=logistic_probs2,truth=IonoTest$V35)
abline(a=0,b=1)
auc.tmp = performance(prediction(logistic_probs2,IonoTest$V35),"auc");
auc.glm = as.numeric(auc.tmp@y.values);
auc.glm 

# Linear Discriminant Analysis (LDA)
library(MASS)
lda.model2= lda(V35~V3+V5+V7+V8+V27, data=IonoTrain)
lda_pred_y2= predict(lda.model2, IonoTest)
table(lda_pred_y2$class,IonoTest$V35)
mean(lda_pred_y2$class == IonoTest$V35)
mean(lda_pred_y2$class != IonoTest$V35)

#CV for Lda
library(sortinghat)
X=cbind(IonoTrain$V3,IonoTrain$V5,IonoTrain$V7,IonoTrain$V8,IonoTrain$V27)
Y=IonoTrain$V35
lda_wpa=function(object,newdata){predict(object,newdata)$class}
errorest_cv(x=X,y=Y,train = lda,classify = lda_wpa,num_folds = 10)

#ROC for Lda
rocplot=function(pred,truth,...)
{
  predob=prediction(pred,truth)
  perf=performance(predob,"tpr","fpr")
  plot(perf,...)
}

rocplot(pred =lda_pred_y2$posterior[,2],truth=IonoTest$V35);
abline(a=0,b=1)
auc.tmp = performance(prediction(lda_pred_y2$posterior[,2],IonoTest$V35),"auc");
auc.lda = as.numeric(auc.tmp@y.values);
auc.lda 

# Quadratic Discriminant Analysis (QDA)

qda.model1 = qda(V35~V3+V5+V7+V8+V27,data=IonoTrain)
qda_pred_y1 = predict(qda.model1, IonoTest)
table(qda_pred_y1$class,IonoTest$V35)
mean(qda_pred_y1$class == IonoTest$V35)
mean(qda_pred_y1$class != IonoTest$V35)
# CV for Qda
qda_wpa=function(object,newdata){predict(object,newdata)$class}
errorest_cv(x=X,y=Y,train = qda,classify = qda_wpa,num_folds = 10)

#ROC for Qda
set.seed(5)
rocplot=function(pred,truth,...)
{
  predob=prediction(pred,truth)
  perf=performance(predob,"tpr","fpr")
  plot(perf,...)
}

rocplot(pred =qda_pred_y1$posterior[,2],truth=IonoTest$V35);
abline(a=0,b=1)
auc.tmp = performance(prediction(qda_pred_y1$posterior[,2],IonoTest$V35),"auc");
auc.qda = as.numeric(auc.tmp@y.values);
auc.qda

#KNN 
library (class)
set.seed (1)
train.X=cbind(IonoTrain$V3,IonoTrain$V5,IonoTrain$V7,IonoTrain$V8,IonoTrain$V27)
test.X=cbind(IonoTest$V3,IonoTest$V5,IonoTest$V7,IonoTest$V8,IonoTest$V27)
train.Y=cbind(IonoTrain$V35)
test.Y=cbind(IonoTest$V35)
knn.pred=knn(train.X,test.X,train.Y,k=3)
table(knn.pred,test.Y)
mean(knn.pred==test.Y)
mean(knn.pred!=test.Y)

#CV for KNN
x=cbind(IonoTrain$V3,IonoTrain$V5,IonoTrain$V7,IonoTrain$V8,IonoTrain$V27)
y=IonoTrain$V35
out= knn.cv(x,y,k=5)
Error = 1 - sum(abs(y == out)) / length(out)
Error

#ROC for KNN
set.seed(5)
rocplot=function(pred,truth,...)
{
  predob=prediction(pred,truth)
  perf=performance(predob,"tpr","fpr")
  plot(perf,...)
}
set.seed(6)
knn.pred=knn(data.frame(train.X),data.frame(test.X),IonoTrain$V35,k=32,prob = TRUE)
rocplot(pred=attr(knn.pred,"prob"),truth = IonoTest$V35)
abline(a=0,b=1)
auc.tmp = performance(prediction(attr(knn.pred,"prob"),IonoTest$V35),"auc");
auc.knn = as.numeric(auc.tmp@y.values);
auc.knn 

# Simple Classification Trees
library (ISLR)
library (tree)
attach(IonoTrain)
tree.Data =tree(V35~V3+V5+V7+V8+V27,IonoTrain)
summary (tree.Data)
plot(tree.Data)
text(tree.Data,pretty =0)
tree.pred=predict(tree.Data,IonoTest,type="class")
table(tree.pred,IonoTest$V35)
mean(tree.pred==IonoTest$V35)
mean(tree.pred!=IonoTest$V35)

#First Cross-Validation to prune the tree optimaly.
set.seed (3)
cv.Data =cv.tree(tree.Data,FUN=prune.misclass )
cv.Data
cv.Data$size
cv.Data$dev
plot(cv.Data)
par(mfrow =c(1,2))
plot(cv.Data$size ,cv.Data$dev ,type="b")
plot(cv.Data$k ,cv.Data$dev ,type="b")

# Pruning. 

prune.data = prune.misclass(tree.Data ,best =3)
plot(prune.data)
text(prune.data,pretty =0)


tree.pred1=predict(prune.data,IonoTest,type="class")
table(tree.pred1,IonoTest$V35)
mean(tree.pred1==IonoTest$V35)
mean(tree.pred1!=IonoTest$V35)


#Bagging. 
library(randomForest)
bag.model=randomForest(V35~V3+V5+V22+I(V26^2)+V27,data=IonoTrain, mtry=5,importance=TRUE)
summary(bag.model)

bag.pred = predict (bag.model ,IonoTest)
plot(bag.pred ,IonoTest$V35)
abline (0,1)
table(bag.pred ,IonoTest$V35)
mean((bag.pred==IonoTest$V35)^2)
mean((bag.pred!=IonoTest$V35)^2)



# Random Forest.


library(caret)
set.seed(56)
rf.model1=randomForest(V35~V3+V5+V7+V8+V27,data=IonoTrain, mtry=3,importance=TRUE)
summary(rf.model1)
rf.pred = predict (rf.model1 ,IonoTest)
plot(rf.pred ,IonoTest$V35)
abline (0,1)
table(rf.pred ,IonoTest$V35)
mean((rf.pred==IonoTest$V35)^2)
mean((rf.pred!=IonoTest$V35)^2)

#ROC for tree

tree.predict=predict(prune.data,IonoTest,type="vector")
rocplot=function(pred,truth,...)
{
  predob=prediction(pred,truth)
  perf=performance(predob,"tpr","fpr")
  plot(perf,...)
}
length(IonoTest$V35)
rocplot(pred=tree.predict[,2],truth=IonoTest$V35);
abline(a=0,b=1)
auc_tmp = performance(prediction(tree.predict[,2],IonoTest$V35),"auc");
auc.rf = as.numeric(auc_tmp@y.values);
auc.rf
#SVC
tune.out=tune(svm,V35~V3+V5+V7+V8+V27,data=IonoTrain ,kernel="linear",ranges=list(cost=c(0.001, 0.01,0.1, 1,5,10)))
summary(tune.out)
#SVM
tune.out=tune(svm, V35~V3+V5+V7+V8+V27, data=IonoTrain, kernel="radial",ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
summary(tune.out)

#Neural Networks
install.packages("neuralnet")
require(neuralnet)
response=ifelse(IonoTrain$V35=="g",1,0)
nn=neuralnet(V35~V3+V5+V7+V8+V27,data=IonoTrain,hidden=2,err.fct="ce",linear.output=FALSE)
nn
plot(nn)
nn$net.result
nn$weights
nn$result.matrix
nn$covariate
nn$net.result[[1]]
nn1=ifelse(nn$net.result[[1]]>0.5,1,0)
miscclassificationError=mean(response!=nn1)
miscclassificationError
OutputvsPred=cbind(response,nn1)
OutputvsPred
ci=confidence.interval(nn,alpha=0.05)
ci
par(mfrow=c(2,2))
gwplot(nn,selected.covariate="V3",min=-1,max=1)
gwplot(nn,selected.covariate="V5",min=-1,max=1)
gwplot(nn,selected.covariate="V7",min=-1,max=1)
gwplot(nn,selected.covariate="V8",min=-1,max=1)
gwplot(nn,selected.covariate="V27",min=-1,max=1)

install.packages("randomForest")
library(randomForest)

# New Model
# PCA
attach(ionosphere)
V35= ifelse(V35 == "b", 0,1)
ionosphere <- data.frame(ionosphere[,-c(2,35)], V35=as.factor(V35))
set.seed(1)
pca=princomp(ionosphere1, cor=TRUE, score= TRUE)
plot(pca,type="lines",lwd=4,col="blue")
summary(pca)

ionosphere1 <- ionosphere[,-34]



#PCA

#LGM on PCA model
PCADATA <- data.frame(pca$scores[,1:7],V35= as.factor(ionosphere[,"V35"]))
model <- glm(V35 ~ .,data= PCADATA[IonoTrain,], family="binomial")
summary(model)
step(model,direction="both",trace=FALSE)
Data2 <- PCADATA[-IonoTrain,c(1:8)]
logistic_probs = predict(model, newdata=Data2, type ="V35")
length(logistic_probs)
logistic_pred_y = rep(0,nrow(IonoTest))
logistic_pred_y[logistic_probs > 0.5] = 1
table(logistic_pred_y,IonoTest$V35)
mean(logistic_pred_y != IonoTest$V35)

# 10-fold cross validation for glm on our PCA model
library(boot)
set.seed(1)
model <- glm(V35 ~ .,data= PCADATA[IonoTrain,], family="binomial")
cv.error.10 = cv.glm(PCADATA[IonoTrain,], model,K=10)$delta[1]
cv.error.10

#LDA on PCA model
library(MASS)
lda.fit = lda(V35 ~ ., data =PCADATA[IonoTrain,])
lda.pred = predict(lda.fit, Data2)
table(lda.pred$class, IonoTest$V35)
mean(lda.pred$class != IonoTest$V35)


#LOOCV cross validation for lda
library(MASS)
lda.fit1 = lda(V35 ~ ., data =PCADATA, subset=IonoTrain, CV= TRUE)
tab <- table(PCADATA[IonoTrain,"V35"], lda.fit1$class)
conCV1 <- rbind(tab[1, ], tab[2, ])
dimnames(conCV1) <- list(Actual = c("No", "Yes"), "Predicted (cv)" = c("No" , "Yes"))
conCV1


#QDA on PCA model
qda.fit = qda(V35 ~ ., data =PCADATA[IonoTrain,])
qda.class = predict(qda.fit, Data2)$class
table(qda.class, IonoTest$V35)
mean(qda.class != IonoTest$V35)

#LOOCV cross validation for QDA on PCA model
library(MASS)
qda.fit1 = qda(V35 ~ ., data = PCADATA[IonoTrain,], CV= TRUE)
tab <- table(PCADATA[IonoTrain,"V35"], qda.fit1$class)
conCV1 <- rbind(tab[1, ], tab[2, ])
dimnames(conCV1) <- list(Actual = c("No", "Yes"), "Predicted (cv)" = c("No" , "Yes"))
conCV1

#KNN on PCA model
library (class)
IonoTraining_V35 = V35[IonoTrain]
knn_pred_padwear = knn(PCADATA[IonoTrain,],PCADATA[-IonoTrain,],IonoTraining_V35,k=1)
#condusion matrix
table(knn_pred_padwear,IonoTest$V35)
mean(knn_pred_padwear!= IonoTest$V35)

#k=10
library (class)
IonoTraining_V35 = V35[IonoTrain]
knn_pred_padwear = knn(PCADATA[IonoTrain,],PCADATA[-IonoTrain,],IonoTraining_V35,k=10)
#condusion matrix
table(knn_pred_padwear,IonoTest$V35)
mean(knn_pred_padwear!= IonoTest$V35)


#k=3
library (class)
IonoTraining_V35 = V35[IonoTrain]
knn_pred_padwear = knn(PCADATA[IonoTrain,],PCADATA[-IonoTrain,],IonoTraining_V35,k=3)
#condusion matrix
table(knn_pred_padwear,IonoTest$V35)
mean(knn_pred_padwear!= IonoTest$V35)

#k=20
library (class)
IonoTraining_V35 = V35[IonoTrain]
knn_pred_padwear = knn(PCADATA[IonoTrain,],PCADATA[-IonoTrain,],IonoTraining_V35,k=20)
#condusion matrix
table(knn_pred_padwear,IonoTest$V35)
mean(knn_pred_padwear!= IonoTest$V35)


#graphical representation:
par(mfrow=c(3,2))
plot(V35, V26)
plot(V35, V27)
plot(V35, V22)
plot(V35, V5)
plot(V35, V3)
plot(V35, V6)

# Simple Classification Trees
install.packages("tree")
library (ISLR)
library (tree)
attach(ionosphere)
tree.Data =tree(V35~V3+V5+V7+V8+V27, IonoTrain)
tree.Data= tree(V35~., PCADATA[IonoTrain,])
summary (tree.Data)

tree.pred=predict(tree.Data,IonoTest,type="class")
tree.pred=predict(tree.Data,PCADATA[-IonoTrain,],type="class")
table(tree.pred,IonoTest$V35)
mean(tree.pred!=IonoTest$V35)



##Cross validation and testing on support vector classifier
set.seed(1)
#All variables
tune.out=tune(svm,V35~.,data=IonoTrain ,kernel="linear",ranges=list(cost=c(0.001, 0.01,0.1, 1,5,10)))
#PCA model
tune.out=tune(svm,V35~.,data=PCADATA[IonoTrain,] ,kernel="linear",ranges=list(cost=c(0.001, 0.01,0.1, 1,5,10)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)


#Testing
ypred=predict(bestmod,IonoTest,type= "V35")
ypred
table(ypred, IonoTest$V35)
mean(ypred != IonoTest$V35)

svmfit=svm(V35~V3+V5+V7+V8+V27, data=IonoTrain, kernel="linear", cost=0.1,scale=FALSE)
svmfit1=svm(V35~V3+V5+V22+V26+V27, data=IonoTrain, kernel="linear", cost=5,scale=FALSE)
svmfit2=svm(V35~., data=PCADATA[IonoTrain,], kernel="linear", cost=5,scale=FALSE)
summary(svmfit1)
ypred=predict(svmfit,IonoTest)
ypred=predict(svmfit2,PCADATA[-IonoTrain,])
table(predict=ypred, truth=IonoTest$V35)
mean(ypred != IonoTest$V35)

#ROC Curve
install.packages("ROCR")
library(ROCR)
rocplot=function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}
svmfit.opt=svm(V35~V3+V5+V22+V26+V27, data=IonoTrain, kernel="linear", cost=0.1,decision.values=T)
fitted=(predict(svmfit.opt,IonoTrain,decision.values=FALSE))
par(mfrow=c(1,2))
rocplot(as.numeric(fitted),ionosphere[IonoTrain,"V35"],main="IonoTraining Data")

svmfit.flex=svm(V35~V3+V5+V22+V26+V27, data=IonoTrain, kernel="linear", cost=10, decision.values=T)
fitted=predict(svmfit.flex,IonoTrain,decision.values=F)
rocplot(as.numeric(fitted),ionosphere[IonoTrain,"V35"],add=T,col="red")
fitted= predict(svmfit.opt,IonoTest,decision.values=F)
rocplot(as.numeric(fitted),IonoTest$V35,main="Test Data")
fitted= predict(svmfit.flex,IonoTest,decision.values=F)
rocplot(as.numeric(fitted),IonoTest$V35,add=T,col="red")
area.svC= performance(prediction(as.numeric(fitted),IonoTest$V35),"auc")
as.numeric(area.svm@y.values)

#PCA model
svmfit.opt=svm(V35~., data=PCADATA[IonoTrain,], kernel="linear", cost=0.1,decision.values=T)
fitted=(predict(svmfit.opt,PCADATA[IonoTrain,],decision.values=FALSE))
par(mfrow=c(1,2))
rocplot(as.numeric(fitted),PCADATA[IonoTrain,"V35"],main="IonoTraining Data")
svmfit.flex=svm(V35~., data=PCADATA[IonoTrain,], kernel="linear", cost=10, decision.values=T)
fitted=predict(svmfit.flex,PCADATA[IonoTrain,],decision.values=F)
rocplot(as.numeric(fitted),PCADATA[IonoTrain,"V35"],add=T,col="red")
fitted= predict(svmfit.opt,PCADATA[-IonoTrain,],decision.values=F)
rocplot(as.numeric(fitted),PCADATA[-IonoTrain,"V35"],main="Test Data")
fitted= predict(svmfit.flex,PCADATA[-IonoTrain,],decision.values=F)
rocplot(as.numeric(fitted),PCADATA[-IonoTrain,"V35"],add=T,col="red")

area.svC= performance(prediction(as.numeric(fitted),PCADATA[-IonoTrain,"V35"]),"auc")
as.numeric(area.svm@y.values)



##Cross validation and testing on SVM

tune.out=tune(svm,V35~.,data=PCADATA[IonoTrain,] ,kernel="radial",ranges=list(cost=c(0.001, 0.01,0.1, 1,5,10),gamma=c(0.5,1,2,3,4)))
summary(tune.out)
bestmod=tune.out$best.model


svmfit=svm(V35~., data=IonoTrain, kernel="radial", gamma=1, cost=1)
summary(svmfit)
svmfit$index
svmfit=svm(V35~., data=IonoTrain, kernel="radial",gamma=1,cost=1e5)

set.seed(1)
#All variables
tune.out=tune(svm, V35~., data=IonoTrain, kernel="radial",ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
#RandomForest
pred=predict(tune.out$best.model,IonoTest)
pred=predict(tune.out$best.model,PCADATA[-IonoTrain,])
table(truth=IonoTest$V35, pred)
truth=IonoTest$V35
mean(IonoTest$V35 != pred)



#ROC Curve for PCA model
svmfit.opt=svm(V35~., data=PCADATA[IonoTrain,], kernel="radial", gamma= 0.5, cost=1,decision.values=T)
fitted=(predict(svmfit.opt,PCADATA[IonoTrain,],decision.values=FALSE))
par(mfrow=c(1,2))
rocplot(as.numeric(fitted),PCADATA[IonoTrain,"V35"],main="IonoTraining Data")
svmfit.flex=svm(V35~., data=PCADATA[IonoTrain,], kernel="radial", gamma= 50, cost=1, decision.values=T)
fitted=predict(svmfit.flex,PCADATA[IonoTrain,],decision.values=F)
rocplot(as.numeric(fitted),PCADATA[IonoTrain,"V35"],add=T,col="red")
fitted= predict(svmfit.opt,PCADATA[-IonoTrain,],decision.values=F)
rocplot(as.numeric(fitted),PCADATA[-IonoTrain,"V35"],main="Test Data")
fitted= predict(svmfit.flex,PCADATA[-IonoTrain,],decision.values=F)
rocplot(as.numeric(fitted),PCADATA[-IonoTrain,"V35"],add=T,col="red")

area.svC= performance(prediction(as.numeric(fitted),PCADATA[-IonoTrain,"V35"]),"auc")
as.numeric(area.svm@y.values)



#neural network on PCA model
install.packages("neuralnet")
require(neuralnet)
response=ifelse(IonoTrain$V35==1,1,0)
response1= ifelse(PCADATA[IonoTrain,]$V35 ==1,1,0)
nn=neuralnet(V35~V3+V5+V22+I(V26^2)+V27,data=IonoTrain,hidden=2,err.fct="ce",linear.output=FALSE)
nn=neuralnet(response1~Comp.1+Comp.2+Comp.3+Comp.4+Comp.5+Comp.6+Comp.7,data=PCADATA[IonoTrain,],hidden=2,err.fct="ce",linear.output=FALSE)
nn=neuralnet(V35~V3+V5+V7+V8+V27,data=IonoTrain,hidden=2,err.fct="ce",linear.output=FALSE)
nn
plot(nn)
nn$net.result
nn$weights
nn$result.matrix
nn$covariate
nn$net.result[[1]]
nn1=ifelse(nn$net.result[[1]]>0.5,1,0)
table(V35, nn1)
miscclassificationError=mean(response!=nn1)
miscclassificationError
OutputvsPred=cbind(V35,nn1)
OutputvsPred
ci=confidence.interval(nn,alpha=0.05)
ci
par(mfrow=c(2,2))
gwplot(nn,selected.covariate="V3",min=-1,max=1)
gwplot(nn,selected.covariate="V5",min=-1,max=1)
gwplot(nn,selected.covariate="V22",min=-1,max=1)
gwplot(nn,selected.covariate="I(V26^2)",min=-1,max=1)
gwplot(nn,selected.covariate="V27",min=-1,max=1)



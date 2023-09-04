setwd ("/Project")

# clear env
rm(list=ls())

# loading up packages
library(nnet)
library(caret)
library(ggplot2)
library(vip)

# set seed
set.seed(4308) # reproducibility

# data frame
df = read.csv("imputeddata.csv", header=TRUE)

# train-test split
splitIndex <- createDataPartition(df$VCPP,p=0.79,list=FALSE) 

df_split <- df[splitIndex,]
df_test <-df[-splitIndex,]

# train-validation split
trainIndex <- createDataPartition(df_split$VCPP,p=0.735,list=FALSE) 

df_train <- df_split[trainIndex,]
df_val <- df_split[-trainIndex,]

#Get the Y variable"
y = df[-splitIndex,"VCPP"]

#Perform the min-max normalization":
maxs = apply(df_train, 2, max)
#find maxima by columns (i.e., for each variable)
mins = apply(df_train, 2, min)
#find minima by columns (i.e., for each variable)

#Do the normalization. Note scale subtracts the "center" parameter 
## and divides by "scale" parameter:
df_train = as.data.frame(scale(df_train, center = mins, scale = maxs - mins))
df_test = as.data.frame(scale(df_test, center = mins, scale = maxs - mins))

# Frequently used lambda values in network training 
## (taken from tensoflowplayground):
dec=c(0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10)

nd=length(dec) 
#no of lambdas, will determine no of iterations in a loop

k=which(colnames(df_test)=="VCPP") 
#find the column index of the Y variable in the data

nnmse=NULL #blank for collecting MSE for different lambdas

#Fit ANN's looping over lambdas:
set.seed(4308) # reproducibility
for(i in 1:nd) {
  nn=nnet(VCPP~., data=df_train, 
          size=10,  maxit=1000, 
          decay=dec[i], linout = TRUE, 
          trace=FALSE, MaxNWts = 10000)
  
  # Predict the test values and un-normalize 
  ## the predictions (don't forget!)
  yhat=predict(nn, df_test) *(maxs[k]-mins[k])+mins[k]
  
  #Compute the test set MSE
  nnmse[i]=summary(lm((yhat-y)^2~1))$coef[1]
}

# results
nnmse 
# [1] 0.34796757 0.09148308 0.06981285 0.05736883 0.03666284 0.02078311 0.01637489 0.01685936 0.01740377 0.01994070

# best nn mse
min(nnmse) # [1] 0.01637489

# variable importance - top 20 graph
plot(vip(nn, num_features=10, geom="col", horizontal=TRUE))

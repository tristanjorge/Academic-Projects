# 4308 MASTER FILE
##START OF RF CODE###############################
rm(list=ls()) #clear environment
library(randomForest)
library(vip)
library(nnet)
library(caret)
library(ggplot2)
library(vip)
library(gbm)
library(dplyr)
library(glmnet)
library(hdm)
library(readr)
library(MLmetrics)
library(lsei)
library(Rcpp)


setwd ("D:/NUS/Y4S3/EC4308 (done)/Project") #sets the working directory that contains the csv file, which is the data downloaded from UCI.
data <- read.csv("Book2.csv", header = TRUE, sep = ",") #collects the file "Book2.csv" from the directory. header = TRUE because headers were pre-added in Excel. since it's a Comma Separated Var file, we specify it in the function. the dataframe is simply named as 'data'.
data[data == "?"] <- NA #there is a lot of missing values labelled "?" in the csv file. rename "?" as NA.
set.seed(4308) #sets a random seed
sapply(data, class) #this checks the classes of features. we want every class to be "numeric" because imputing NA values with the RF function only works with "numeric" classes.

#the following code replaces numeric data with class "character" as class "numeric".
data$LemasSwornFT.numeric <- as.numeric(data$LemasSwornFT.numeric)
data$OtherPerCap.numeric <- as.numeric(data$OtherPerCap.numeric)
data$LemasSwFTPerPop.numeric <- as.numeric(data$LemasSwFTPerPop.numeric)
data$LemasSwFTFieldOps.numeric <- as.numeric(data$LemasSwFTFieldOps.numeric)
data$LemasSwFTFieldPerPop.numeric <- as.numeric(data$LemasSwFTFieldPerPop.numeric)
data$LemasTotalReq.numeric <- as.numeric(data$LemasTotalReq.numeric)
data$LemasTotReqPerPop.numeric  <- as.numeric(data$LemasTotReqPerPop.numeric )
data$PolicReqPerOffic.numeric <- as.numeric(data$PolicReqPerOffic.numeric)
data$PolicPerPop.numeric <- as.numeric(data$PolicPerPop.numeric)
data$PolicBudgPerPop.numeric <- as.numeric(data$PolicBudgPerPop.numeric)
data$LemasGangUnitDeploy.numeric <- as.numeric(data$LemasGangUnitDeploy.numeric)
data$PolicOperBudg.numeric <- as.numeric(data$PolicOperBudg.numeric)
data$LemasPctPolicOnPatr.numeric <- as.numeric(data$LemasPctPolicOnPatr.numeric)
data$PolicCars.numeric <- as.numeric(data$PolicCars.numeric)
data$PolicAveOTWorked.numeric <- as.numeric(data$PolicAveOTWorked.numeric)
data$OfficAssgnDrugUnits.numeric <- as.numeric(data$OfficAssgnDrugUnits.numeric)
data$PctPolicAsian.numeric <- as.numeric(data$PctPolicAsian.numeric)
data$PctPolicBlack.numeric <- as.numeric(data$PctPolicBlack.numeric)
data$RacialMatchCommPol.numeric <- as.numeric(data$RacialMatchCommPol.numeric)
data$NumKindsDrugsSeiz.numeric <- as.numeric(data$NumKindsDrugsSeiz.numeric)
data$PctPolicMinor.numeric <- as.numeric(data$PctPolicMinor.numeric)
data$PctPolicHisp.numeric <- as.numeric(data$PctPolicHisp.numeric)
data$PctPolicWhite.numeric <- as.numeric(data$PctPolicWhite.numeric)

names(data)[names(data) == 'ViolentCrimesPerPop.numeric'] <- 'VCPP' #renames the goal variable for simplicity

data.imputed <- randomForest::rfImpute(VCPP ~ ., data=data, iter=6) #impute values for NAs.
#in the first argument, we specify the goal variable VCPP to be predicted by all the other predictive features.
#in the second argument, we specify what dataframe to use. since the dataframe was named as "data", we specify data=data.
#in the third argument, we specify how many RFs rfImpute should grow to estimate the NA values.
#in theory, 4-6 iterations should be fine.
#save the imputed dataframe as "data.imputed"

#generates an imputed CSV to be used by all group members and saves it as 'imputedDataWithSeed4308.csv'
#write.csv(data.imputed,"C:/Users/whatt/Desktop/Desktop R working dir\\imputedDataWithSeed4308.csv", row.names = FALSE)

# set seed
set.seed(4308) # reproducibility

# data frame
df = data.imputed

# train-test split
trainIndex <- createDataPartition(df$VCPP,p=0.80,list=FALSE) 

df_train <- df[trainIndex,]
df_test <-df[-trainIndex,] #has all 123 features including the goal feature

df_test122 <- df_test[,-1] #only the x variables, excluding the goal feature
df_train122 <- df_train[,-1] #only the x variables, excluding the goal feature

df_testgoal <- df_test[,1, drop=FALSE] #only the y variable of test set
df_traingoal <- df_train[,1, drop=FALSE] #only the y variable of train set

####################start of LASSO code##########

# setting grid for lambda (starts denser, getting sparser)
grid = 10^seq(10, -2, length = 100)

# Apply LASSO with different values of lambda
#all sets were defined as data frames. glmnet only takes in matrices. matrix conversion is done.
set.seed(4308)
lasso_mod <- glmnet(data.matrix(df_train122), data.matrix(df_traingoal), alpha = 1, lambda = grid)
plot(lasso_mod)

sum_abs_coef = c()
lambdas = c()
for (i in 1:ncol(coef(lasso_mod))){
  lambdas = c(lambdas, lasso_mod$lambda[i])
  coefs = coef(lasso_mod)[-1,i]
  sum_abs_coef = c(sum_abs_coef,
                   sum(abs(coefs^2))
  )
}

plot(x = lambdas, y = sum_abs_coef, xlim = c(0, 300),
     xlab = "values of lambdas",
     ylab = "sum of the absolute values of the coefficients")

# LASSO 10-fold CV with Default Grid 
set.seed(4308)
cv10fold_lasso = cv.glmnet(data.matrix(df_train122), data.matrix(df_traingoal), alpha = 1)
plot(cv10fold_lasso)

plot(cv10fold_lasso$lambda,cv10fold_lasso$cvm,
     main="10-fold CV default settings - LASSO",
     xlab="Lambda", ylab="CV MSE")

minlambda_l = cv10fold_lasso$lambda[which.min(cv10fold_lasso$cvm)]
minlambda_l

lasso_pred <- predict(lasso_mod, s = minlambda_l, newx = data.matrix(df_test122))
mean((lasso_pred - data.matrix(df_testgoal))^2)

# LASSO 10-fold CV with User's Grid
set.seed(4308)
newgrid = seq(0, 0.03, length.out = 10000)
cv10fold_lasso_ug= cv.glmnet(data.matrix(df_train122), data.matrix(df_traingoal), alpha = 1,
                             lambda=newgrid)

plot(cv10fold_lasso_ug$lambda,cv10fold_lasso_ug$cvm,
     main="10-fold CV - LASSO - User's Grid",
     xlab="Lambda", ylab="CV MSE")

minlambda_l_ug = cv10fold_lasso_ug$lambda.min
minlambda_l_ug

lasso_pred <- predict(lasso_mod, s = minlambda_l_ug, newx = data.matrix(df_test122))
MSELASSO = mean((lasso_pred - data.matrix(df_testgoal))^2)

#################end of LASSO code

#############start of ELASTIC NET code#######

grid = 10^seq(10, -2, length = 100)

al=0.5
elas_net <- glmnet(data.matrix(df_train122), data.matrix(df_traingoal), alpha = al, lambda = grid)
plot(elas_net)

set.seed(4308)
elas_net_cv  = cv.glmnet(data.matrix(df_train122), data.matrix(df_traingoal), alpha = al, lambda=grid)
plot(elas_net_cv $lambda,elas_net_cv $cvm,
     main="10-fold CV user-defined grid",
     xlab="Lambda", ylab="CV MSE",
     xlim = c(0, 0.5),
     ylim = c(-1,1))

lamda_min_cv_el= elas_net_cv$lambda.min
lamda_min_cv_el

elas_net_cv$cvm[which(elas_net_cv$lambda == elas_net_cv$lambda.min)]

predicts_elcv_oos <- predict(elas_net_cv,
                             s = lamda_min_cv_el,
                             newx = data.matrix(df_test122))
mse_elcv_oos = mean((predicts_elcv_oos - data.matrix(df_testgoal))^2)
MSEElastic = mse_elcv_oos

#######################################################

#7 random forests with different selections of m
set.seed(4308)
bagging <- randomForest(VCPP~., data=data.matrix(df_train), proximity=TRUE, mtry=122)
RFdefault <-randomForest(VCPP~., data=data.matrix(df_train), proximity=TRUE)
randomforesttuned <- tuneRF(data.matrix(df_train122), data.matrix(df_traingoal), ntreeTry=500, stepFactor=1.5, improve=0.05, trace=FALSE, plot=FALSE, doBest=TRUE)

#displays results 
# bagging
# RFdefault
baggingpredict = predict(bagging, newdata = data.matrix(df_test122))
MSEbagging = MSE(baggingpredict, data.matrix(df_testgoal))

RFdefaultpredict =  predict(RFdefault, newdata = data.matrix(df_test122))
MSERFdefault = MSE(RFdefaultpredict, data.matrix(df_testgoal))

TunedRFpredict = predict(randomforesttuned, newdata = data.matrix(df_test122))
MSETunedRandomForest = MSE(TunedRFpredict, data.matrix(df_testgoal))

#generates variable importance plots for the 8 random forests
vip(bagging, geom="col")
vip(RFdefault, geom="col")


#################################################END OF RF CODE#########

#################################################START OF ANN AND GBM CODE#########



#Perform the min-max normalization":
maxs = apply(df_train, 2, max) 
#find maxima by columns (i.e., for each variable)
mins = apply(df_train, 2, min) 
#find minima by columns (i.e., for each variable)

#Do the normalization. Note scale subtracts the "center" parameter 
## and divides by "scale" parameter:
df_train_ANN = as.data.frame(scale(df_train, center = mins, scale = maxs - mins))
df_test_ANN = as.data.frame(scale(df_test, center = mins, scale = maxs - mins))

y = df[-trainIndex,"VCPP"]
# Frequently used lambda values in network training 
## (taken from tensoflowplayground):
dec=c(0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10)

nd=length(dec) 
#no of lambdas, will determine no of iterations in a loop

k=which(colnames(df_test_ANN)=="VCPP") 
#find the column index of the Y variable in the data

nnmse=NULL #blank for collecting MSE for different lambdas

#Fit ANN's looping over lambdas:
set.seed(4308) # reproducibility
for(i in 1:nd) {
  nn=nnet(VCPP~., data=df_train_ANN, 
          size=10,  maxit=1000, 
          decay=dec[i], linout = TRUE, 
          trace=FALSE, MaxNWts = 10000)
  
  # Predict the test values and un-normalize 
  ## the predictions (don't forget!)
  yhat=predict(nn, df_test_ANN)*(maxs[k]-mins[k])+mins[k]
  
  #Compute the test set MSE
  nnmse[i]=summary(lm((yhat-y)^2~1))$coef[1]
}

# results
nnmse 
# [1] 64.82866183  0.08789873  0.08717548  0.04689725  0.02779873  0.02017240  0.01639020  0.01660042  0.01734771
# [10]  0.01902379

# best nn mse
MSENeural = min(nnmse) # [1] 0.0163902
print(MSENeural)

# variable importance - top 10 graph
plot(vip(nn, num_features=10, geom="col", horizontal=TRUE))

#############start of GBM

# train GBM model
set.seed(4308) # reproducibility

gbm.fit <- gbm(
  formula = VCPP ~ .,
  distribution = "gaussian",
  data = df_train,
  n.trees = 10000,
  interaction.depth = 1,
  shrinkage = 0.001,
  train.fraction = 1,
  cv.folds = 10,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)

print(gbm.fit)
print(min(gbm.fit$cv.error)) # [1] 0.01926316
gbm.perf(gbm.fit, method = "cv") # [1] 10000

# tune GBM model

set.seed(4308) # reproducibility

gbm.fit2 <- gbm(
  formula = VCPP ~ .,
  distribution = "gaussian",
  data = df_train,
  n.trees = 5000,
  interaction.depth = 3,
  shrinkage = 0.01,
  train.fraction = 1,
  cv.folds = 10,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)

print(gbm.fit2)
print(min(gbm.fit2$cv.error)) # [1] 0.01893943
gbm.perf(gbm.fit2, method = "cv") # [1] 1472

# create hyperparameter grid to assess different combinations of hyperparameters
hyper_grid <- expand.grid(
  shrinkage = c(.01, 0.5, .1), # determine optimal learning rate
  interaction.depth = c(3, 5, 7), # highest level of variable interactions
  n.minobsinnode = c(5, 10, 15), # vary min. no. of obs. in tree terminal nodes
  bag.fraction = c(.65, .8, 1), # introduce stochastic gradient descent
  optimal_trees = 0,               # a place to dump results
  min_MSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid) # 81

# randomize data
set.seed(4308)
random_index <- sample(1:nrow(df_train), nrow(df_train))
random_df_train <- df_train[random_index, ]

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(4308)
  
  # train model
  gbm.tune <- gbm(
    formula = VCPP ~ .,
    distribution = "gaussian",
    data = random_df_train,
    n.trees = 2500,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = 0.7,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_MSE[i] <- min(gbm.tune$valid.error)
}

hyper_grid %>% 
  dplyr::arrange(min_MSE) %>%
  head(10)
#   shrinkage interaction.depth n.minobsinnode bag.fraction optimal_trees    min_MSE
# 1       0.01                 7              5         0.65           668 0.01707578
# 2       0.01                 7             10         0.65           670 0.01719975
# 3       0.01                 7             15         0.65          1237 0.01728900
# 4       0.10                 3             15         0.65           222 0.01731691
# 5       0.10                 3             10         0.65           179 0.01736683
# 6       0.01                 5              5         0.65           775 0.01739482
# 7       0.01                 7             10         0.80          1341 0.01742191
# 8       0.01                 5             15         0.65          1237 0.01747583
# 9       0.01                 5             10         0.65          1083 0.01751727
# 10      0.01                 3             10         0.65          2002 0.01753708

# results of gbm.tune
# 1. few of the top models used stumps (interaction.depth = 1); there are likely
# some important interactions that the deeper trees are able to capture
# 2. adding a stochastic component with bag.fraction < 1 seems to help; 
# there may be some local minimas in our loss function gradient

# train final GBM model using top model from gbm.tune
set.seed(4308) # reproducibility

gbm.fit.final <- gbm(
  formula = VCPP ~ .,
  distribution = "gaussian",
  data = df_train,
  n.trees = 668,
  interaction.depth = 7,
  shrinkage = 0.01,
  n.minobsinnode = 5,
  bag.fraction = .65,
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)

# Visualization of results
vip::vip(gbm.fit.final, num_features=10, geom="col", horizontal=TRUE)

# predict values for test data
predGBM <- predict(gbm.fit.final, n.trees = gbm.fit.final$n.trees, df_test)

# results - square of RMSE to obtain OOB MSE
MSEgbm <- caret::RMSE(predGBM, df_test$VCPP)**2 # [1] 0.01690916
print(MSEgbm)

#######################end of ANN and GBM code



######################forecast combinations (WIP)######
# 
# bagging_predc            <- predict(bagging, newdata=data.matrix(df_val122))
# 
# RFdefault_predc          <- predict(RFdefault, newdata=data.matrix(df_val122))  
# 
# cv10fold_lasso_ug_predc  <- predict(cv10fold_lasso_ug, s = minlambda_l_ug, newx = data.matrix(df_val122))
# 
# elas_net_cv_predc        <- predict(elas_net_cv, s=lamda_min_cv_el, newx = data.matrix(df_val122))
# 
# nn_predc                 <- predict(nn, newdata=data.matrix(df_val122))
# 
# gbm.fit.final_predc      <- predict(gbm.fit.final, newdata=df_val122, n.trees=gbm.fit.final$n.trees)
# 
# fmatu=cbind(bagging_predc, 
#             RFdefault_predc, 
#             cv10fold_lasso_ug_predc, 
#             elas_net_cv_predc, 
#             nn_predc, 
#             gbm.fit.final_predc)
# 
# gru = lsei(fmatu, data.matrix(df_valgoal), c=rep(1,6), d=1, e=diag(6), f=rep(0,6))

# comb_pred = gru[1]*bagging_predc + gru[2]*RFdefault_predc
# combination <- predict(comb_pred, newdata=data.matrix(df_test122))
# MSE(combination, data.matrix(df_testgoal))

# #tristan
# comb_predu= gru[2]*RFdefault_predc + gru[3]*cv10fold_lasso_ug_predc 
# + gru[4]*elas_net_cv_predc + gru[6]*gbm.fit.final_predc
# MSE(data.matrix(df_testgoal), comb_predu)
# 
# #redefine X matrix for forecasts
# fmatb=cbind(rep(1,nrow(cv10fold_lasso_ug_predc)), #column of ones (for the constant)
#             bagging_predc, 
#             RFdefault_predc, 
#             cv10fold_lasso_ug_predc, 
#             elas_net_cv_predc, 
#             nn_predc, 
#             gbm.fit.final_predc)
# 
# temp=diag(7)
# temp[1,1]=0
# 
# #Find the GR weights under constraints, bu with constant in the regression:
# grb=lsei(fmatb, data.matrix(df_valgoal), c=c(0,rep(1,6)), d=1, e=temp, f=rep(0,7))
# 
# #From the forecasts using nonzero weights:
# combpredb=grb[3]*RFdefault_predc + grb[4]*cv10fold_lasso_ug_predc 
# + grb[5]*elas_net_cv_predc + grb[7]*gbm.fit.final_predc
# MSE(data.matrix(df_testgoal),combpredb) #check MSE
# 
# # unrestricted weights: no constraints, no constant
# grunr=lsei(fmatu,data.matrix(df_valgoal))
# 
# #Form combined forecast (almost all weights nonzero, 
# ## so do vector product to sum):
# combpredur=cbind(bagging_predc, RFdefault_predc, cv10fold_lasso_ug_predc, 
#                  elas_net_cv_predc, nn_predc, gbm.fit.final_predc)%*%grunr
# MSE(data.matrix(df_testgoal),combpredur)
# 
# # unrestricted weights: no constraints, include constant
# grunrc=lsei(fmatb,data.matrix(df_valgoal))
# combpredurc=cbind(rep(1,nrow(cv10fold_lasso_ug_predc)),
#                   bagging_predc, 
#                   RFdefault_predc, 
#                   cv10fold_lasso_ug_predc, 
#                   elas_net_cv_predc, 
#                   nn_predc, 
#                   gbm.fit.final_predc)%*%grunrc
# MSE(data.matrix(df_testgoal),combpredurc)

print(MSELASSO)
print(MSEElastic)
print(MSEbagging)
print(MSERFdefault)
print(MSETunedRandomForest)
print(MSENeural)
print(MSEgbm)
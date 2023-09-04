setwd ("D:/NUS/Y4S3/EC4308 (done)/Project")

# clear env
rm(list=ls())

# loading up packages
library(caret)
library(gbm)
library(ggplot2)
library(dplyr)

# reproducibility
set.seed(4308)

df = read.csv("imputeddata.csv", header=TRUE)

#train-test split
splitIndex <- createDataPartition(df$VCPP,p=0.80,list=FALSE) 

df_train <- df[splitIndex,]
df_test <-df[-splitIndex,]

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
print(min(gbm.fit$cv.error)) # [1] 0.01920487
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
print(min(gbm.fit2$cv.error)) # [1] 0.01887967
gbm.perf(gbm.fit2, method = "cv") # [1] 1337

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
    n.trees = 6000,
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
  hyper_grid$min_MSE[i] <- min(gbm.tune$valid.error) # removed sqrt to calc min_MSE instead
}

hyper_grid %>% 
  dplyr::arrange(min_MSE) %>%
  head(10)
# shrinkage interaction.depth n.minobsinnode bag.fraction optimal_trees    min_MSE
# 1       0.01                 3              5         0.65           856 0.01788218
# 2       0.01                 5              5         0.65           806 0.01789895
# 3       0.01                 5             10         0.65           973 0.01793114
# 4       0.01                 3             10         0.65          2313 0.01795804
# 5       0.01                 7             15         0.65           867 0.01800540
# 6       0.01                 3             15         0.65           792 0.01803390
# 7       0.01                 7             10         0.65           539 0.01804744
# 8       0.01                 7              5         0.65           716 0.01810843
# 9       0.01                 3             15         0.80          1270 0.01814434
# 10      0.10                 5              5         0.65            63 0.01815974

# results of gbm.tune
# 1. none of the top models used a learning rate of 0.01
# 2. few of the top models used stumps (interaction.depth = 1); there are likely
# some important interactions that the deeper trees are able to capture
# 3. adding a stochastic component with bag.fraction < 1 seems to help; 
# there may be some local minimas in our loss function gradient

# train final GBM model using top model from gbm.tune
set.seed(4308) # reproducibility

gbm.fit.final <- gbm(
  formula = VCPP ~ .,
  distribution = "gaussian",
  data = df_train,
  n.trees = 590,
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
pred <- predict(gbm.fit.final, n.trees = gbm.fit.final$n.trees, df_test)

# results - square of RMSE to obtain OOB MSE
MSEgbm <- caret::RMSE(pred, df_test$VCPP)**2 # [1] 0.01662394
print(MSEgbm) #[1] 0.01706002
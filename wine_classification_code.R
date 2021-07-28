### Code Summary: ##############################################################
# The purpose of this project is to create an accurate wine quality classifier.
# 
### Database: ##################################################################
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties.
# In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
#
### Basic data sets: ###########################################################
# Download red and white wine data sets
# Creation of train and test sets

url  <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"
repo <- "http://cran.us.r-project.org"

# Required packages:
if(!require(tidyverse))     install.packages("tidyverse",    repos = repo)
if(!require(caret))         install.packages("caret",        repos = repo)
if(!require(data.table))    install.packages("data.table",   repos = repo)
if(!require(purrr))         install.packages("purrr",        repos = repo)
if(!require(randomForest))  install.packages("randomForest", repos = repo)
if(!require(h2o))           install.packages("h2o",          repos = repo)
if(!require(Boruta))        install.packages("Boruta",     repos = repo)

library(tidyverse)
library(caret)
library(data.table)
library(purrr)
library(randomForest)
library(h2o)
library(Boruta)

dl <- tempfile()
# Download red_wine data set
download.file(paste(url,"/winequality-red.csv",sep = ""), dl)
red_wine <- read.csv(dl, sep =  ";")
# Download white_wine data set  
download.file(paste(url,"/winequality-white.csv",sep = ""), dl)
white_wine <- read.csv(dl, sep =  ";")

# Create red_wine general and validation set
set.seed(2021, sample.kind = "Rounding")

red_test_index <- createDataPartition(red_wine$quality, p = 0.1, list = FALSE)

red_general <- red_wine[-red_test_index,]
red_validation <- red_wine[red_test_index,]

# Create white_wine general and validation set
set.seed(2021, sample.kind = "Rounding")

white_test_index <- createDataPartition(white_wine$quality, p = 0.1, list = FALSE)

white_general <- red_wine[-white_test_index,]
white_validation <- red_wine[white_test_index,]

rm(dl, red_wine, red_test_index, white_wine, white_test_index, repo, url)
gc()

### Data exploration: ##########################################################
# Explore data looking for a better understanding of the data sets.
# 
# Useful information in the winequality.names file:
#
#   - These datasets can be viewed as classification or regression tasks.
#   - The classes are ordered and not balanced 
#     (e.g. there are munch more normal wines than excellent or poor ones).
#   - Outlier detection algorithms could be used to detect the few excellent 
#     or poor wines. 
#   - Several of the attributes may be correlated, thus it makes sense to apply
#     some sort of feature selection.
#

### CORRELATION TEST
cor_limit <- 0.6

## red wine correlation
red_cor <- round(cor(red_general),2)

red_cor_lim_ind <- lapply(apply(red_cor < -cor_limit | red_cor > cor_limit & red_cor < 1,
                            1, which), names)

red_cor_lim_ind <- red_cor_lim_ind[lengths(red_cor_lim_ind) > 0]

red_cor_list <- lapply(c("fixed.acidity","free.sulfur.dioxide"), function(x){
  select(all_of(x),.data = as.data.frame(red_cor[names(red_cor_lim_ind),
                                                 names(red_cor_lim_ind)])) %>% 
  filter_all(all_vars(. < -cor_limit | . > cor_limit & . < 1))
})

# NOT SUFFICIENT CORRELATION(aprox +-.67), although:
# fixed.acidity may substitute citric.acid, density, and pH (cor .67, .67, -.68)
# free.sulfur.dioxide may substitute total.sulfur.dioxide   (cor .65)

## white wine correlation
white_cor <- round(cor(white_general),2)

w_cor_lim_ind <- lapply(apply(white_cor < -cor_limit | white_cor > cor_limit & white_cor < 1,
                            1, which), names)

w_cor_lim_ind <- w_cor_lim_ind[lengths(w_cor_lim_ind) > 0]

white_cor_list <- lapply(c("fixed.acidity","free.sulfur.dioxide"), function(x){
  select(all_of(x),.data = as.data.frame(white_cor[names(w_cor_lim_ind),
                                                 names(w_cor_lim_ind)])) %>% 
    filter_all(all_vars(. < -cor_limit | . > cor_limit & . < 1))
})

# NOT SUFFICIENT CORRELATION(aprox +-.67), although:
# fixed.acidity may substitute citric.acid, density, and pH (cor .68, .67, -.69)
# free.sulfur.dioxide may substitute total.sulfur.dioxide   (cor .67)
# greater correlation than red wine data

## Red Wine Boruta attribute importance
set.seed(2021, sample.kind = "Rounding")

rg_boruta <- Boruta(as.factor(quality) ~ ., data = red_general, doTrace = 2, maxRuns = 500)
rg_boruta

## White Wine Boruta attribute importance
set.seed(2021, sample.kind = "Rounding")

wg_boruta <- Boruta(as.factor(quality) ~ ., data = white_general, doTrace = 2, maxRuns = 500)
wg_boruta

rm(red_cor, red_cor_lim_ind, white_cor, w_cor_lim_ind, cor_limit, rg_boruta, wg_boruta)
gc()

### Data Cleaning: #############################################################
### ANOMALY DETECTION TEST
# Isolation forest with h2o
threshold <- .98

# RED WINE!
#Initializing h2o on the localhost
localH2O <- h2o.init(ip="localhost", port = 54321, 
                     startH2O = TRUE, nthreads=-1)

setwd(getwd())
# Making the quality column, a factor
red_general[,12] <- as.factor(red_general[,12])
h2o_rg <- as.h2o(red_general) # making a data set readable to h2o models

rg_isoforest <- h2o.isolationForest(training_frame = h2o_rg,
                                    sample_rate = 0.1,
                                    max_depth = 50, # max depth of trees
                                    ntrees = 100, # number of trees
                                    seed = 2021)

# Predicting based on the isolation forest
rg_score <- h2o.predict(rg_isoforest, h2o_rg)

# Score limit based on the threshold and the predict score
rg_scoreLimit <- round(quantile(as.vector(rg_score$predict), threshold), 4)

# Bind the "level of abnormality" of each row to the general data set
red_general <- cbind(RowScore = round(as.vector(rg_score$predict), 4), red_general)

# Select the rows higher than the threshold
rg_anomalies <- red_general[red_general$RowScore > rg_scoreLimit,]

# Delete the bind rows
red_general <- red_general[,2:13]
rg_anomalies <- rg_anomalies[,2:13]

# Proportion of anomalies per quality factor
prop.table(table(rg_anomalies$quality))

rm(h2o_rg, rg_anomaly, rg_dl, rg_isoforest, rg_score, rg_scoreLimit)
gc()

# WHITE WINE!
# Making the quality column, a factor
white_general[,12] <- as.factor(white_general[,12])
h2o_w <- as.h2o(white_general) # making a data set readable to h2o models

w_isoforest <- h2o.isolationForest(training_frame = h2o_w,
                                   sample_rate = 0.1,
                                   max_depth = 50, # max depth of trees
                                   ntrees = 100, # number of trees
                                   seed = 2021)

# Predicting based on the isolation forest
w_score <- h2o.predict(w_isoforest, h2o_w)

# Score limit based on the threshold and the predict score
w_scoreLimit <- round(quantile(as.vector(w_score$predict), threshold), 4)

# Bind the "level of abnormality" of each row to the general data set
white_general <- cbind(RowScore = round(as.vector(w_score$predict), 4), white_general)

# Select the rows higher than the threshold
wg_anomalies <- white_general[white_general$RowScore > w_scoreLimit,]

# Delete the bind rows
white_general <- white_general[,2:13]
wg_anomalies <- wg_anomalies[,2:13]

# Proportion of anomalies per quality factor
prop.table(table(wg_anomalies$quality))

rm(threshold, h2o_w, w_anomaly, w_dl, w_isoforest, w_score, w_scoreLimit)
gc()

### Experiment: ################################################################
# Experimenting machine learning methods 
# 

### RED WINE!!
set.seed(2021, sample.kind = "Rounding")
rex_index <- createDataPartition(red_general$quality, p = .2, list = FALSE)

rex_train <- red_general[-rex_index,]
rex_test <- red_general[rex_index,]

# Remove anomalies
rex_train <- anti_join(rex_train, rg_anomalies)

## Random forest TEST
l <- seq(1,500,5)
rex_rf_acc_test <- sapply(l, function(x){
  set.seed(2021, sample.kind = "Rounding")
  
  fit <- randomForest(quality ~ .,
                      data = rex_train,
                      importance = TRUE, # Access importance in the method
                      proximity = TRUE, # Proximity will be calculated
                      ntree=x) # Number of trees
  confusionMatrix(predict(fit, rex_test),
                  as.factor(rex_test$quality))$overall["Accuracy"]
  
})
# Select the best ntree and accuracy
rex_sel_l <- l[which.max(rex_rf_acc_test)]
rex_rf_acc <- max(rex_rf_acc_test)

## Deep Learning TEST

h2o_rex_train <- as.h2o(rex_train)
h2o_rex_test  <- as.h2o(rex_test)

rex_dl <- h2o.deeplearning(x = names(h2o_rex_train[,1:11]), # predictors
                           y = names(h2o_rex_train[,12]), # to be predicted
                           training_frame = h2o_rex_train,
                           model_id = "rex_tdl",
                           activation = "Tanh",
                           epochs = 200,
                           balance_classes = TRUE,
                           max_after_balance_size = 6, # inflate the sample size
                           # up to 6 times
                           hidden = c(200,200,100), # arbitrary number of layers
                           seed = 2021)

# accuracy based on the confusion matrix in the test set
rex_dl_acc <- 1- h2o.performance(rex_dl,h2o_rex_test)@metrics$cm$table[["Error"]][7]

## h2o gbm TEST
rex_gbm <- h2o.gbm(x = names(h2o_rex_train[,1:11]),
                   y = names(h2o_rex_train[,12]),
                   training_frame = h2o_rex_train,
                   model_id = "rex_gbm",
                   ntrees = 700, # Number of trees
                   balance_classes = TRUE,
                   seed = 2021)

# Accuracy based on the confusion matrix in the test set
rex_gbm_acc <- 1 - h2o.performance(rex_gbm,h2o_rex_test)@metrics$cm$table[["Error"]][7]

### WHITE WINE!!
set.seed(2021, sample.kind = "Rounding")
wex_index <- createDataPartition(white_general$quality, p = .2, list = FALSE)

wex_train <- white_general[-wex_index,]
wex_test <- white_general[wex_index,]

# Remove anomalies
wex_train <- anti_join(wex_train, wg_anomalies)

## Random forest TEST
l <- seq(1,500,5)
wex_rf_acc_test <- sapply(l, function(x){
  set.seed(2021, sample.kind = "Rounding")
  
  fit <- randomForest(quality ~ .,
                      data = wex_train,
                      importance = TRUE, # Access importance in the method
                      proximity = TRUE, # Proximity will be calculated
                      ntree=x) # Number of trees
  confusionMatrix(predict(fit, wex_test),
                  as.factor(wex_test$quality))$overall["Accuracy"]
  
})
# Select the best ntree and accuracy
wex_sel_l <- l[which.max(wex_rf_acc_test)]
wex_rf_acc <- max(wex_rf_acc_test)

## Deep Learning TEST

h2o_wex_train <- as.h2o(wex_train)
h2o_wex_test  <- as.h2o(wex_test)

wex_dl <- h2o.deeplearning(x = names(h2o_wex_train[,1:11]), # predictors
                           y = names(h2o_wex_train[,12]), # to be predicted
                           training_frame = h2o_wex_train,
                           model_id = "wex_tdl",
                           activation = "Tanh",
                           epochs = 200,
                           balance_classes = TRUE,
                           max_after_balance_size = 6, # inflate the sample size
                           # up to 6 times
                           hidden = c(200,200,100), # arbitrary number of layers
                           seed = 2021)

# accuracy based on the confusion matrix in the test set
wex_dl_acc <- 1 - h2o.performance(wex_dl,h2o_wex_test)@metrics$cm$table[["Error"]][7]

## h2o gbm TEST

wex_gbm <- h2o.gbm(x = names(h2o_wex_train[,1:11]),
                   y = names(h2o_wex_train[,12]),
                   training_frame = h2o_wex_train,
                   model_id = "wex_gbm",
                   ntrees = 700,
                   balance_classes = TRUE,
                   seed = 2021)
wex_gbm_acc <- 1 - h2o.performance(wex_gbm,h2o_wex_test)@metrics$cm$table[["Error"]][7]

### ACCURACY TABLE
accuracy_tbl <- data.frame(Method     = c("Random Forest", "Deep Learning", "GBM"),
                           Red_Wine   = c(rex_rf_acc, rex_dl_acc, rex_gbm_acc),
                           White_Wine = c(wex_rf_acc, wex_dl_acc, wex_gbm_acc))
accuracy_tbl

### Final validation: ##########################################################
# Using the prefered method [Random Forest] on the validation set :) 
# 

## RED WINE

# Remove anomalies
red_general <- anti_join(red_general, rg_anomalies)

# Random Forest
set.seed(2021, sample.kind = "Rounding")
red_fit <- randomForest(quality ~ .,
                    data = red_general,
                    importance = TRUE,
                    proximity = TRUE, 
                    ntree=rex_sel_l)

red_val_acc <- confusionMatrix(predict(red_fit, red_validation),
                               as.factor(red_validation$quality))$overall["Accuracy"]

## WHITE WINE

# Remove anomalies
white_general <- anti_join(white_general, rg_anomalies)

# Random Forest
set.seed(2021, sample.kind = "Rounding")
wed_fit <- randomForest(quality ~ .,
                        data = white_general,
                        importance = TRUE,
                        proximity = TRUE, 
                        ntree=wex_sel_l)

wed_val_acc <- confusionMatrix(predict(wed_fit, white_validation),
                               as.factor(white_validation$quality))$overall["Accuracy"]

## FINAL ACCURACY TABLE

val_acc_tbl <- data.frame(Random_Forest = c("Expected", "Real"),
                          Red_Wine      = c(rex_rf_acc, red_val_acc),
                          White_Wine    = c(wex_rf_acc, wed_val_acc))
val_acc_tbl

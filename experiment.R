### Code Summary: ##############################################################
# This script has the intent of testing method for future validation!
#
### Database: ##################################################################
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties.
# In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
#
### Basic data sets: ###########################################################
# Download red and white wine data sets
# Creation of train and test sets

repo <- "http://cran.us.r-project.org"

# Required packages:
if(!require(tidyverse))     install.packages("tidyverse",    repos = repo)
if(!require(caret))         install.packages("caret",        repos = repo)
if(!require(data.table))    install.packages("data.table",   repos = repo)
if(!require(Boruta))        install.packages("Boruta",       repos = repo)
if(!require(randomForest))  install.packages("randomForest", repos = repo)
if(!require(h2o))           install.packages("h2o",          repos = repo)
if(!require(e1071))         install.packages("e1071",        repos = repo)


library(tidyverse)
library(caret)
library(data.table)
library(Boruta)
library(randomForest)
library(h2o)
library(e1071)

# Check if the red and white wine required data sets are available, otherwise
# downloading them :)
if(!exists("red_wine_general") &
   !exists("red_wine_validation") &
   !exists("white_wine_general") &
   !exists("white_wine_validation")){
  
  dl <- tempfile()
  # Download red_wine data set
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", dl)
  red_wine <- read.csv(dl, sep =  ";")
  # Download white_wine data set  
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", dl)
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
  
  rm(dl, red_wine, red_test_index, white_wine, white_test_index)
}

gc()

# RMSE function
rmse <- function(true, predicted) {
  sqrt(mean((true - predicted)^2))
}

### Data Cleaning: #############################################################

set.seed(2021, sample.kind = "Rounding")

rg_index <- createDataPartition(red_general$quality, p = 0.1, list = FALSE)

# Useful variables:
# fixed.acidity | volatile.acidity | citric.acid | sulphates | alcohol
rg_train <- red_general[-rg_index,]
rg_test  <- red_general[rg_index,]

# Proportion
prop.table(table(rg_train$quality))

# Cleaning outliners [quality 3, ?4?, and 8]
# Proven after factor selection not quite useful :/
rgc_train <- rg_train %>% filter(quality != 3 &
                                 quality != 4 &
                                 quality != 8)

# Recursive Feature Elimination:
set.seed(2021, sample.kind = "Rounding")

ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 10)

results <- rfe(rg_train[,1:11], rg_train[,12], sizes = c(1:11), rfeControl = ctrl)

# Feature Selection through Random Forest and Boruta:
set.seed(2021, sample.kind = "Rounding")

boruta <- Boruta(as.factor(quality) ~ ., data = rgc_train, doTrace = 2, maxRuns = 500)
boruta

set.seed(2021, sample.kind = "Rounding")

l <- seq(1,500,10)


tree_acc <- sapply(l, function(x){
  set.seed(2021, sample.kind = "Rounding")
  
  fit <- randomForest(as.factor(quality) ~ ., data = rg_train, ntree=x)
  confusionMatrix(predict(fit, rg_test),
                  as.factor(rg_test$quality))$overall["Accuracy"]

})

sel_l <- l[which.max(tree_acc)]
sel_l
max(tree_acc)

ref_l <- seq(sel_l - 20, sel_l + 20,1)

tree_ref_acc <- sapply(ref_l, function(x){
  set.seed(2021, sample.kind = "Rounding")
  
  fit <- randomForest(as.factor(quality) ~ ., data = rg_train, ntree=x)
  confusionMatrix(predict(fit, rg_test),
                  as.factor(rg_test$quality))$overall["Accuracy"]
  
})

ref_sel_l <- ref_l[which.max(tree_ref_acc)]
ref_sel_l
max(tree_ref_acc)


set.seed(2021, sample.kind = "Rounding")

fit_rf <- randomForest(as.factor(quality) ~ ., 
                       data = rg_train,
                       ntree=ref_sel_l)

confusionMatrix(predict(fit_rf, rg_test),
                as.factor(rg_test$quality))$overall["Accuracy"]

### DEEP LEARNING TEST

localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

y = names(rg_train)[12]
x = names(rg_train)[1:11]

h20_rgtrain <- rg_train
h20_rgtest  <- rg_test

h20_rgtrain[,y] <- as.factor(h20_rgtrain[,y])
h20_rgtest[,y]  <- as.factor(h20_rgtest[,y])

setwd(getwd())
write.csv(h20_rgtrain, "h20_rgtrain.csv")
write.csv(h20_rgtest, "h20_rgtest.csv")

h20_rgtrain <- h2o.importFile("h20_rgtrain.csv")
h20_rgtest  <- h2o.importFile("h20_rgtest.csv")

hfit <- h2o.deeplearning(x=x, 
                         y=y, 
                         training_frame=h20_rgtrain, 
                         validation_frame=h20_rgtest, 
                         distribution = "auto",
                         activation = "RectifierWithDropout",
                         hidden = c(10,10,10,10),
                         input_dropout_ratio = 0.2,
                         l1 = 1e-5,
                         epochs = 50)

hfit_gbm <- h2o.gbm(x=x, 
                    y=y, 
                    training_frame=h20_rgtrain, 
                    validation_frame=h20_rgtest, 
                    sample_rate = 0.7,
                    seed = 2021)

hfit_auto <- h2o.automl(y=y, 
                    training_frame=h20_rgtrain, 
                    max_runtime_secs = 120,
                    seed = 2021)
print(hfit)


################################################################################
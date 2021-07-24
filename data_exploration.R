### Code Summary: ##############################################################
# This script has the intent of exploring the trainig data sets!
#
### Database: ##################################################################
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties.
# In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
#
### Basic data sets: ###########################################################
# Download red and white wine data sets
# Creation of train and test sets
# Required functions for further exploration

repo <- "http://cran.us.r-project.org"

# Required packages:
if(!require(tidyverse))      install.packages("tidyverse",      repos = repo)
if(!require(caret))          install.packages("caret",          repos = repo)
if(!require(data.table))     install.packages("data.table",     repos = repo)
if(!require(h2o))           install.packages("h2o",          repos = repo)

library(tidyverse)
library(caret)
library(data.table)
library(h2o)

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

### Themes!
# Red wine theme xD
rw_color <- "#64242e"
theme_rw <- function(){
  theme_bw(base_size = 12, base_family = "times") %+replace%
    theme(panel.background = element_rect("#F1F1E9"),
          panel.grid = element_line("#DFDFDF"))
}

# White wine theme >w<
ww_color <- "#9BC888"
theme_ww <- function(){
  theme_bw(base_size = 12, base_family = "times") %+replace%
    theme(panel.background = element_rect("#EDF9E7"),
          panel.grid = element_line("#CBD5C6"))
}

### Plot generator function:
# Since we must analyse 11 plots per wine type, it is easier to use a function 
# and automatize it!
#

# Graphic titles
var_names <- colnames(red_general[,-12]) %>%
  str_replace("\\.", " ") %>%
  str_replace("\\.", " ") %>%
  str_to_title()

# Function
plot_gen <- function(x,data){
  p <- data %>% ggplot(aes_string("quality",
                                  colnames(data)[x],
                                  group = "quality")) +
    ggtitle(paste("Quality X", var_names[x])) +
    scale_x_continuous(name = "Quality", breaks = c(3:8)) +
    ylab(var_names[x])
  
  if(nrow(data) == nrow(red_general)){
    p + geom_boxplot(fill  = rw_color, color = "black") + theme_rw()
  } else if(nrow(data) == nrow(white_general)){
    p + geom_boxplot(fill  = ww_color, color = "black") + theme_ww()
  }
}

### Data exploration: ##########################################################
# Explore data looking for a better understanding of the data sets.
# 
# url: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names 
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

### Red wine set!!!
# Visual distribution of ratings:
red_general %>% ggplot(aes(quality)) +
                geom_histogram(bins = 6, fill = rw_color) +
                ggtitle("Distribution of Ratings per Red Wine") +
                scale_x_continuous(name = "Quality", breaks = c(3:8)) +
                ylab("Number of Wines") +
                theme_rw()

# Visualization of quality correlation and outliers:
lapply(c(1:11), plot_gen, data = red_general)

# Correlation data
red_cor <- cor(red_general)[,12]

# Proportion
prop.table(table(red_general$quality))

# Anomaly detection with h2o ISOLATION FOREST

localH2O <- h2o.init(ip="localhost", port = 54321, 
                     startH2O = TRUE, nthreads=-1)

setwd(getwd())
red_general[,12] <- as.factor(red_general[,12])
write.csv(red_general, "red_general.csv")
h2o_rg <- h2o.importFile("red_general.csv")

rg_dl <- h2o.deeplearning(x=names(red_general)[1:11],
                          training_frame = h2o_rg,
                          autoencoder = TRUE,
                          activation = "Tanh",
                          hidden = c(10,10,10,10),
                          epochs = 100,
                          seed = 2021)
anomaly <- h2o.anomaly(rg_dl, h2o_rg)
row_anomaly <- which(anomaly > 0.99)
      
isoforest <- h2o.isolationForest(training_frame = h2o_rg,
                                 sample_rate = 0.1,
                                 max_depth = 20,
                                 ntrees = 50)

tred_general <- red_general

score <- h2o.predict(isoforest, h2o_rg)
result_pred <- as.vector(score$predict)
threshold <- .97
scoreLimit = round( quantile( result_pred, threshold ), 4 )
tred_general = cbind( RowScore = round( result_pred, 4 ), tred_general )
anomalies = tred_general[ tred_general$RowScore > scoreLimit, ]

### White wine set!!!
# Visual distribution of ratings:
white_general %>% ggplot(aes(quality)) +
                  geom_histogram(bins = 6, fill = ww_color) +
                  ggtitle("Distribution of Ratings per White Wine") +
                  scale_x_continuous(name = "Quality", breaks = c(3:8)) +
                  ylab("Number of Wines") +
                  theme_ww()

# Visualization of correlation and outliers:
lapply(c(1:11), plot_gen, data = white_general)

# Correlation data
white_cor <- cor(white_general)

################################################################################
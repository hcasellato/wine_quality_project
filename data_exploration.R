### Code Summary: ##############################################################
# This script has the intent of exploring the trainig set
#
### Basic data sets: ###########################################################
# Download train and test datasets
# Note: this process could take a couple of minutes

if(!require(tidyverse))      install.packages("tidyverse",      repos = repo)
if(!require(caret))          install.packages("caret",          repos = repo)
if(!require(data.table))     install.packages("data.table",     repos = repo)

library(tidyverse)
library(caret)
library(data.table)


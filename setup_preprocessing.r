

# ============================================================================
# PACKAGE INSTALLATION
# ============================================================================

install.packages("ranger")
install.packages("caret")
install.packages("data.table")
install.packages("caTools")
install.packages("pROC")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("neuralnet")

# ============================================================================
# LIBRARY IMPORTS
# ============================================================================
library(ranger)
library(caret)
library(data.table)
library(caTools)
library(pROC)
library(rpart)
library(rpart.plot)
library(neuralnet)

# ============================================================================
# DATA LOADING AND EXPLORATION
# ============================================================================

# Load the data from file
file_data <- read.csv('creditcard.csv')

# Display basic information about the dataset
cat("Dataset Overview:\n")
cat("================\n")
print(head(file_data))

cat("\nColumn Names:\n")
print(names(file_data))

cat("\nClass Distribution:\n")
print(table(file_data$Class))

cat("\nClass Summary:\n")
print(summary(file_data$Class))

cat("\nTotal Number of Rows:", nrow(file_data), "\n")

# ============================================================================
# DATA VISUALIZATION
# ============================================================================

# Create plots to visualize the data distribution
par(mfrow = c(1, 2))
x = 1:nrow(file_data)
y = file_data$Amount
my_factor <- factor(file_data$Amount)

# Scatterplot of transaction amounts
plot(x, y, main = "Scatterplot - Original Amount", xlab="Count", ylab="Amount")

# Barplot of amount factors
plot(my_factor, main = "Barplot - Original Amount", xlab="Factor Amount", ylab="Count")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

# Scale the Amount column to normalize the data
file_data$Amount <- scale(file_data$Amount)

# Remove the Time column (first column) as it's not needed for modeling
data <- file_data[, -c(1)]

cat("\nData preprocessing completed.\n")
cat("Amount column has been scaled.\n")
cat("Time column has been removed.\n")

# Verify the scaling with new plots
par(mfrow = c(1, 2))
y_scaled = data$Amount
my_factor_scaled <- factor(data$Amount)

# Scatterplot of scaled amounts
plot(x, y_scaled, main = "Scatterplot - Scaled Amount", xlab="Count", ylab="Amount")

# Barplot of scaled amount factors
plot(my_factor_scaled, main = "Barplot - Scaled Amount", xlab="Factor Amount", ylab="Count")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets (80-20 split)
data_sample <- sample.split(data$Class, SplitRatio = 0.80)
data.train_data <- subset(data, data_sample == TRUE)
data.test_data <- subset(data, data_sample == FALSE)

# Verify the splits
cat("\nData Split Summary:\n")
cat("==================\n")
cat("Training set size:", nrow(data.train_data), "rows\n")
cat("Testing set size:", nrow(data.test_data), "rows\n")
cat("Split ratio:", round(nrow(data.train_data) / nrow(data), 3), ":", round(nrow(data.test_data) / nrow(data), 3), "\n")

# Save the preprocessed data for use in other scripts
save(data, data.train_data, data.test_data, file = "preprocessed_data.RData")
cat("\nPreprocessed data saved to 'preprocessed_data.RData'\n")
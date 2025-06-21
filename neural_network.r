

# ============================================================================
# SETUP
# ============================================================================
library(neuralnet)
library(caTools)

# Load preprocessed data
load("preprocessed_data.RData")

cat("Neural Network Model for Credit Card Fraud Detection\n")
cat("====================================================\n")

# ============================================================================
# DATA PREPARATION FOR NEURAL NETWORK
# ============================================================================

cat("Preparing smaller dataset for neural network training...\n")

# Create a smaller dataset for neural network training (due to computational constraints)
set.seed(123)
data_sample <- sample.split(data.test_data$Class, SplitRatio = 0.80)
data.train_data_sm <- subset(data.test_data, data_sample == TRUE)
data.test_data_sm <- subset(data.test_data, data_sample == FALSE)

cat("Small training set size:", nrow(data.train_data_sm), "rows\n")
cat("Small testing set size:", nrow(data.test_data_sm), "rows\n")

# ============================================================================
# MODEL TRAINING
# ============================================================================

cat("\nTraining neural network model...\n")
cat("Architecture: Input layer -> Hidden layer 1 (5 neurons) -> Hidden layer 2 (2 neurons) -> Output layer\n")

# Fit the neural network
# Note: Using a smaller threshold and fewer hidden layers for faster computation
nn <- neuralnet(Class ~ ., 
                data = data.train_data_sm, 
                hidden = c(5, 2), 
                linear.output = FALSE, 
                threshold = 0.01,
                stepmax = 1e6)  # Increase max steps if needed

cat("Neural network training completed.\n")

# ============================================================================
# MODEL PREDICTIONS
# ============================================================================

cat("Making predictions on training data (for validation)...\n")

# Get the results on training data
nn.results <- compute(nn, data.train_data_sm[, -ncol(data.train_data_sm)])  # Exclude target variable
results <- data.frame(actual = data.train_data_sm$Class, 
                     prediction = nn.results$net.result)

# ============================================================================
# MODEL EVALUATION
# ============================================================================

cat("Evaluating neural network performance...\n")

# Round the predictions to get binary classification
roundedresults <- sapply(results, round, digits = 0)
roundedresultsdf <- data.frame(roundedresults)

# Create confusion matrix
confusion_matrix <- table(Actual = roundedresultsdf$actual, 
                         Predicted = roundedresultsdf$prediction)

cat("\nConfusion Matrix:\n")
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix) * 100
cat("\nAccuracy:", round(accuracy, 2), "%\n")

# Calculate detailed metrics if possible
if (nrow(confusion_matrix) == 2 && ncol(confusion_matrix) == 2) {
  TP <- confusion_matrix[2, 2]  # True Positives
  TN <- confusion_matrix[1, 1]  # True Negatives
  FP <- confusion_matrix[1, 2]  # False Positives
  FN <- confusion_matrix[2, 1]  # False Negatives
  
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  specificity <- TN / (TN + FP)
  
  cat("Precision:", round(precision, 4), "\n")
  cat("Recall (Sensitivity):", round(recall, 4), "\n")
  cat("Specificity:", round(specificity, 4), "\n")
  cat("F1-Score:", round(f1_score, 4), "\n")
}

# ============================================================================
# NEURAL NETWORK VISUALIZATION
# ============================================================================

cat("\nGenerating neural network plot...\n")

# Plot the neural network structure
plot(nn, 
     main = "Neural Network Structure for Credit Card Fraud Detection",
     rep = "best")

# ============================================================================
# ADDITIONAL PREDICTIONS ON TEST SET
# ============================================================================

if (nrow(data.test_data_sm) > 0) {
  cat("\nMaking predictions on test set...\n")
  
  # Predict on actual test set
  nn.test_results <- compute(nn, data.test_data_sm[, -ncol(data.test_data_sm)])
  test_results <- data.frame(actual = data.test_data_sm$Class, 
                            prediction = nn.test_results$net.result)
  
  # Round test predictions
  rounded_test_results <- sapply(test_results, round, digits = 0)
  rounded_test_df <- data.frame(rounded_test_results)
  
  # Test set confusion matrix
  test_confusion_matrix <- table(Actual = rounded_test_df$actual, 
                                Predicted = rounded_test_df$prediction)
  
  cat("Test Set Confusion Matrix:\n")
  print(test_confusion_matrix)
  
  # Test set accuracy
  test_accuracy <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix) * 100
  cat("Test Set Accuracy:", round(test_accuracy, 2), "%\n")
}

# ============================================================================
# MODEL INFORMATION
# ============================================================================

cat("\nNeural Network Model Information:\n")
cat("=================================\n")
cat("Number of weights:", length(nn$weights[[1]]), "\n")
cat("Error achieved:", nn$result.matrix[1], "\n")
cat("Steps taken:", nn$result.matrix[3], "\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save the model and results
neural_network_results <- list(
  model = nn,
  training_results = results,
  training_confusion_matrix = confusion_matrix,
  training_accuracy = accuracy,
  predictions = roundedresultsdf$prediction,
  actual = roundedresultsdf$actual
)

# Add test results if available
if (exists("test_results")) {
  neural_network_results$test_results <- test_results
  neural_network_results$test_confusion_matrix <- test_confusion_matrix
  neural_network_results$test_accuracy <- test_accuracy
}

save(neural_network_results, file = "neural_network_results.RData")
cat("\nResults saved to 'neural_network_results.RData'\n")

cat("\n", rep("=", 50), "\n")
cat("Neural Network Analysis Complete\n")
cat(rep("=", 50), "\n")
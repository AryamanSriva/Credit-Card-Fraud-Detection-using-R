

# ============================================================================
# SETUP
# ============================================================================
library(rpart)
library(rpart.plot)

# Load preprocessed data
load("preprocessed_data.RData")

cat("Decision Tree Model for Credit Card Fraud Detection\n")
cat("===================================================\n")

# ============================================================================
# MODEL TRAINING
# ============================================================================

cat("Training decision tree model...\n")

# Build the decision tree model
decisionTree_model <- rpart(Class ~ ., data.train_data, method = 'class')

cat("Model training completed.\n\n")

# ============================================================================
# MODEL PREDICTIONS
# ============================================================================

cat("Making predictions on test data...\n")

# Get class predictions
predicted_val <- predict(decisionTree_model, data.test_data, type = 'class')

# Get probability predictions
probability <- predict(decisionTree_model, data.test_data, type = 'prob')

# ============================================================================
# MODEL EVALUATION
# ============================================================================

# Create confusion matrix (truth table)
cf_matrix <- table(Actual = data.test_data$Class, Predicted = predicted_val)
cat("Confusion Matrix:\n")
print(cf_matrix)

# Calculate accuracy
accuracy <- sum(diag(cf_matrix)) / sum(cf_matrix) * 100
cat("\nAccuracy:", round(accuracy, 2), "%\n")

# Calculate detailed metrics
TP <- cf_matrix[2, 2]  # True Positives
TN <- cf_matrix[1, 1]  # True Negatives
FP <- cf_matrix[1, 2]  # False Positives
FN <- cf_matrix[2, 1]  # False Negatives

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)
specificity <- TN / (TN + FP)

cat("Precision:", round(precision, 4), "\n")
cat("Recall (Sensitivity):", round(recall, 4), "\n")
cat("Specificity:", round(specificity, 4), "\n")
cat("F1-Score:", round(f1_score, 4), "\n")

# ============================================================================
# MODEL VISUALIZATION
# ============================================================================

cat("\nGenerating decision tree visualization...\n")

# Plot the decision tree
par(mfrow = c(1, 1))
rpart.plot(decisionTree_model, 
           main = "Decision Tree for Credit Card Fraud Detection",
           extra = 2,
           under = TRUE,
           faclen = 0)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

cat("\nFeature Importance:\n")
cat("==================\n")

# Get variable importance
importance <- decisionTree_model$variable.importance
if (!is.null(importance)) {
  # Sort by importance
  importance_sorted <- sort(importance, decreasing = TRUE)
  cat("Top 10 Most Important Features:\n")
  print(head(importance_sorted, 10))
  
  # Plot feature importance
  par(mar = c(5, 8, 4, 2))
  barplot(head(importance_sorted, 10), 
          horiz = TRUE, 
          las = 1,
          main = "Top 10 Feature Importance - Decision Tree",
          xlab = "Importance Score")
  par(mar = c(5, 4, 4, 2))  # Reset margins
} else {
  cat("No variable importance information available.\n")
}

# ============================================================================
# MODEL DETAILS
# ============================================================================

cat("\nModel Details:\n")
cat("==============\n")
print(decisionTree_model)

cat("\nModel Complexity Parameter (CP) Table:\n")
print(decisionTree_model$cptable)

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save the model and results
decision_tree_results <- list(
  model = decisionTree_model,
  predictions = predicted_val,
  probabilities = probability,
  confusion_matrix = cf_matrix,
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  specificity = specificity,
  f1_score = f1_score,
  feature_importance = importance
)

save(decision_tree_results, file = "decision_tree_results.RData")
cat("\nResults saved to 'decision_tree_results.RData'\n")

cat("\n", rep("=", 50), "\n")
cat("Decision Tree Analysis Complete\n")
cat(rep("=", 50), "\n")
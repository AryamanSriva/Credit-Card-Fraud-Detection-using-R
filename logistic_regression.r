

# ============================================================================
# SETUP
# ============================================================================
library(pROC)

# Load preprocessed data
load("preprocessed_data.RData")

cat("Logistic Regression Model for Credit Card Fraud Detection\n")
cat("=========================================================\n")

# ============================================================================
# MODEL TRAINING
# ============================================================================

# Fit the logistic regression model
cat("Training logistic regression model...\n")
Logistic_Model <- glm(Class ~ ., data.train_data, family = binomial())

cat("Model training completed.\n\n")

# ============================================================================
# MODEL SUMMARY
# ============================================================================

cat("Model Summary:\n")
cat("==============\n")
print(summary(Logistic_Model))

# ============================================================================
# MODEL DIAGNOSTICS
# ============================================================================

cat("\nGenerating diagnostic plots...\n")

# Create diagnostic plots for the logistic regression model
par(mfrow = c(2, 2))
plot(Logistic_Model, main = "Logistic Regression Diagnostics")

# Reset plot parameters
par(mfrow = c(1, 1))

# ============================================================================
# MODEL PREDICTIONS
# ============================================================================

cat("Making predictions on test data...\n")

# Make predictions on the test set
predicted_probabilities <- predict(Logistic_Model, data.test_data, type = "response")

# Convert probabilities to binary predictions (threshold = 0.5)
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)

# ============================================================================
# MODEL EVALUATION
# ============================================================================

# Create confusion matrix
confusion_matrix <- table(Actual = data.test_data$Class, Predicted = predicted_classes)
cat("\nConfusion Matrix:\n")
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix) * 100
cat("\nAccuracy:", round(accuracy, 2), "%\n")

# Calculate other metrics
TP <- confusion_matrix[2, 2]  # True Positives
TN <- confusion_matrix[1, 1]  # True Negatives
FP <- confusion_matrix[1, 2]  # False Positives
FN <- confusion_matrix[2, 1]  # False Negatives

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1-Score:", round(f1_score, 4), "\n")

# ============================================================================
# ROC CURVE ANALYSIS
# ============================================================================

cat("\nGenerating ROC curve...\n")

# Create and plot ROC curve
roc_curve <- roc(data.test_data$Class, predicted_probabilities, plot = TRUE, col = "blue")
title("ROC Curve - Logistic Regression")

# Calculate AUC
auc_value <- auc(roc_curve)
cat("AUC (Area Under Curve):", round(auc_value, 4), "\n")

# Add AUC to the plot
text(0.6, 0.4, paste("AUC =", round(auc_value, 4)), col = "blue", cex = 1.2)

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save the model and results
logistic_results <- list(
  model = Logistic_Model,
  predictions = predicted_classes,
  probabilities = predicted_probabilities,
  confusion_matrix = confusion_matrix,
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  f1_score = f1_score,
  auc = auc_value
)

save(logistic_results, file = "logistic_regression_results.RData")
cat("\nResults saved to 'logistic_regression_results.RData'\n")

cat("\n" , rep("=", 50), "\n")
cat("Logistic Regression Analysis Complete\n")
cat(rep("=", 50), "\n")
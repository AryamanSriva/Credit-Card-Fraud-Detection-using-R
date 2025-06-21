
cat("Credit Card Fraud Detection Analysis\n")
cat("=====================================\n")
cat("Starting comprehensive fraud detection analysis...\n\n")

# Check if required data file exists
if (!file.exists("creditcard.csv")) {
  stop("Error: creditcard.csv file not found in the current directory.")
}

# ============================================================================
# STEP 1: DATA PREPROCESSING
# ============================================================================

cat("STEP 1: Data Preprocessing and Setup\n")
cat("=====================================\n")

tryCatch({
  source("setup_and_preprocessing.R")
  cat("✓ Data preprocessing completed successfully\n\n")
}, error = function(e) {
  cat("✗ Error in data preprocessing:", e$message, "\n")
  stop("Cannot proceed without preprocessed data.")
})

# ============================================================================
# STEP 2: LOGISTIC REGRESSION
# ============================================================================

cat("STEP 2: Logistic Regression Analysis\n")
cat("=====================================\n")

tryCatch({
  source("logistic_regression.R")
  cat("✓ Logistic regression analysis completed successfully\n\n")
}, error = function(e) {
  cat("✗ Error in logistic regression analysis:", e$message, "\n")
  cat("Continuing with other models...\n\n")
})

# ============================================================================
# STEP 3: DECISION TREE
# ============================================================================

cat("STEP 3: Decision Tree Analysis\n")
cat("===============================\n")

tryCatch({
  source("decision_tree.R")
  cat("✓ Decision tree analysis completed successfully\n\n")
}, error = function(e) {
  cat("✗ Error in decision tree analysis:", e$message, "\n")
  cat("Continuing with other models...\n\n")
})

# ============================================================================
# STEP 4: NEURAL NETWORK
# ============================================================================

cat("STEP 4: Neural Network Analysis\n")
cat("================================\n")

tryCatch({
  source("neural_network.R")
  cat("✓ Neural network analysis completed successfully\n\n")
}, error = function(e) {
  cat("✗ Error in neural network analysis:", e$message, "\n")
  cat("Skipping neural network model...\n\n")
})

# ============================================================================
# STEP 5: MODEL COMPARISON
# ============================================================================

cat("STEP 5: Model Comparison and Final Analysis\n")
cat("============================================\n")

tryCatch({
  source("model_comparison.R")
  cat("✓ Model comparison completed successfully\n\n")
}, error = function(e) {
  cat("✗ Error in model comparison:", e$message, "\n")
  cat("Individual model results are still available.\n\n")
})

# ============================================================================
# EXECUTION SUMMARY
# ============================================================================

cat("EXECUTION SUMMARY\n")
cat("=================\n")

# Check which result files were created
result_files <- c(
  "preprocessed_data.RData",
  "logistic_regression_results.RData", 
  "decision_tree_results.RData",
  "neural_network_results.RData",
  "model_comparison_results.RData"
)

cat("Generated Result Files:\n")
for (file in result_files) {
  if (file.exists(file)) {
    cat("✓", file, "\n")
  } else {
    cat("✗", file, "(not created)\n")
  }
}

cat("\nAnalysis Complete!\n")
cat("==================\n")

# Final recommendations
cat("\nNext Steps:\n")
cat("• Review individual model result files for detailed analysis\n")
cat("• Load model_comparison_results.RData for comparative insights\n")
cat("• Consider ensemble methods for improved performance\n")
cat("• Implement the best performing model in production\n")
cat("• Set up monitoring and regular retraining processes\n")

# Display session info for reproducibility
cat("\nSession Information:\n")
cat("====================\n")
print(sessionInfo())
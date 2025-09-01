# Credit Card Fraud Detection 

This project aims to predict fraudulent credit card transactions using ML

The dataset contains 284,807 transactions, of which 492 are identified as fraudulent. Given the highly imbalanced nature of the dataset, appropriate handling is required before model building.

## Understanding and Defining Fraud

Credit card fraud involves dishonest actions and behaviors aimed at obtaining information without the account holder's authorization for financial gain. Among various fraud methods skimming which involves duplicating information from the card's magnetic strip is the most common. Other methods include:

- Manipulation or alteration of genuine cards
- Creation of counterfeit cards
- Use of stolen or lost credit cards
- Fraudulent telemarketing activities

## Data Dictionary

The dataset can be downloaded using this [link](https://www.kaggle.com/mlg-ulb/creditcardfraud).

This dataset includes credit card transactions made by European cardholders over two days in September 2013. Out of 284,807 transactions, 492 are fraudulent. The dataset is highly unbalanced with fraudulent transactions accounting for 0.172% of the total. To maintain confidentiality, Principal Component Analysis (PCA) has been applied. Besides 'time' and 'amount', all other features (V1 to V28) are PCA-derived components. The 'time' feature indicates the seconds elapsed between the first and subsequent transactions. The 'amount' feature represents the transaction amount. The 'class' feature is the target variable, where 1 indicates fraud and 0 indicates non-fraud.

## Project Pipeline

The project pipeline can be summarized in the following steps:

- **Data Understanding:** Load the data and understand the features present. This will help in selecting the features for the final model.
- **Exploratory Data Analysis (EDA):** Perform univariate and bivariate analyses, followed by feature transformations if necessary. Given that Gaussian variables are used, Z-scaling may not be required. However, check for any skewness in the data and address it if necessary.
- **Train/Test Split:** Perform a train/test split to assess model performance on unseen data. Use k-fold cross-validation, choosing an appropriate k value to ensure the minority class is represented in the test folds.
- **Model Building/Hyperparameter Tuning:** Try different models and fine-tune their hyperparameters to achieve desired performance. Explore various sampling techniques to improve the model.
- **Model Evaluation:** Evaluate the models using appropriate metrics. Since the data is imbalanced, it is crucial to accurately identify fraudulent transactions. Choose an evaluation metric that reflects this business objective.

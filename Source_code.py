import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Load the datasets
X = pd.read_csv("training_set_features.csv") # Load the training features
y = pd.read_csv("training_set_labels.csv") # Load the training labels 
z = pd.read_csv("training_set_labels.csv") # Load the training labels (for another target variable)
final = pd.read_csv("test_set_features.csv") # Load the test features

# Remove respondent_id, employment_industry, employment_occupation and health_insurance from X 
X = X.drop(['respondent_id', 'employment_industry', 'employment_occupation', 'health_insurance'], axis=1)

# Remove respondent_id, seasonal_vaccine from y
y = y.drop(['respondent_id', 'seasonal_vaccine'], axis=1)

# Remove respondent_id, seasonal_vaccine from z 
z= z.drop(['respondent_id', 'h1n1_vaccine'], axis=1)

# Extracting the respondent_id column from the final 
respondent_id = final['respondent_id']

# Remove respondent_id, employment_industry, employment_occupation and health_insurance from final dataframe
final = final.drop(['respondent_id', 'employment_industry', 'employment_occupation', 'health_insurance'], axis=1)

# Define numerical and categorical features
numerical_features = X.select_dtypes(include=['float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define numerical and categorical transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(max_iter=12, tol=0.001, random_state=42)),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine numerical and categorical transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define stratified k-fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define function to evaluate model
def evaluate_model(X_data, y_data):
    
    # Compute scale_pos_weight based on the class distribution in the training data
    scale_pos_weight = (len(y_data) - y_data.sum()) / y_data.sum()
    
    # Define the BaggingClassifier with XGBClassifier as the base estimator
    model_xgb = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, max_depth=7,
                              min_child_weight=15, eta=0.07, reg_lambda=0.15,
                              objective="binary:logistic", eval_metric='auc', random_state=42)
    model_bagging = BaggingClassifier(estimator=model_xgb, n_estimators=50, random_state=42, 
                                      max_samples=0.8,max_features=0.9, bootstrap=True)

    roc_auc_scores = []
 
    # Perform stratified k-fold taining and testing
    for train_index, test_index in skf.split(X_data, y_data):
        X_train_skf, X_test_skf = X_data[train_index], X_data[test_index]
        y_train_skf, y_test_skf = y_data[train_index], y_data[test_index]

        # Fit the model
        model_bagging.fit(X_train_skf, y_train_skf)

        # Predict on the test set
        y_pred_skf = model_bagging.predict(X_test_skf)

        # Evaluate the model
        y_pred_proba = model_bagging.predict_proba(X_test_skf)[:, 1]  # Probability of positive class
        roc_auc = roc_auc_score(y_test_skf, y_pred_proba, average="macro")
        roc_auc_scores.append(roc_auc)
        
    # Calculate mean roc_auc scores across all fold
    mean_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
    
    return model_bagging, mean_roc_auc

# Preprocess X
X_processed = preprocessor.fit_transform(X)

# Evaluate model for y
print("Evaluation for y:")
model_y, roc_auc_y = evaluate_model(X_processed, y.values.ravel())
print("Mean ROC AUC Score for h1n1_vaccine =", roc_auc_y)

# Evaluate model for z
print("\nEvaluation for z:")
model_z, roc_auc_z = evaluate_model(X_processed, z.values.ravel())
print("Mean ROC AUC Score for seasonal_vaccine =", roc_auc_z)

# Calculate final mean roc_auc score for both y and z
final_mean_roc_auc = (roc_auc_y + roc_auc_z) / 2
print("\nOverall ROC AUC Score =", final_mean_roc_auc)

# Transform final
final_processed = preprocessor.transform(final)

# Predict probabilities for final 
h1n1_vaccine_probs = model_y.predict_proba(final_processed)[:, 1]
seasonal_vaccine_probs = model_z.predict_proba(final_processed)[:, 1]

# Round the probabilities to one decimal place
h1n1_vaccine = np.round(h1n1_vaccine_probs, 1)
seasonal_vaccine = np.round(seasonal_vaccine_probs, 1)

# Create DataFrame for predictions
predictions_df = pd.DataFrame({
    'respondent_id': respondent_id,
    'h1n1_vaccine': h1n1_vaccine,
    'seasonal_vaccine': seasonal_vaccine
})

# Save results to CSV file
predictions_df.to_csv("results.csv", index=False)

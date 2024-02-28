
#  Predicting Probabilities of H1N1 and Seasonal Vaccination Uptake

This project focuses on predicting the probabilities of individuals receiving h1n1 and seasonal flu vaccinations using machine learning techniques. It leverages data from the 2009 National H1N1 Flu Survey (NHFS) conducted by CDC. The goal is  to inform public health strategies and improve vaccine uptake rates.


## Model Used
BaggingClassifier with XGBoostClassifier as the base estimator.
## Key Points
1. Addressed class imbalance in both of the target variables.
2. Two separate models are instantiated using  BaggingClassifier with XGBoostClassifier as the base estimator for each target variable.
3. Implemented hyperparameter tuning using Stratified RandomizedSearchCV and Stratified GridSearchCV.
4. The best hyperparameters are directly used in the [code](https://github.com/Dream-Falls/AnalyticaX/blob/main/source_code.ipynb).
5. Utilizes ROC AUC for evaluation within each model.
Additionally, we calculated an overall ROC AUC by combining both models' predictions, providing a comprehensive assessment of performance.
## Report

This [report](https://github.com/Dream-Falls/AnalyticaX/blob/main/report.pdf) demonstrates predictive models for H1N1 and seasonal flu vaccination probabilities, covering exploratory data analysis, preprocessing, model initialization, hyperparameter tuning, and evaluation metrics for informed public health strategies.
## Results
The predicted probabilities for the two target variables of the 'test_set_features' dataset are stored [here](https://github.com/Dream-Falls/AnalyticaX/blob/main/results.csv).

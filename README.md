

#  Predicting Probabilities of  H1N1 and Seasonal Vaccination Uptake

This project focuses on predicting the probabilities of individuals receiving H1N1 and seasonal influenza vaccinations using machine learning techniques. It leverages data from the 2009 National H1N1 Flu Survey (NHFS) conducted by CDC. The goal is  to inform public health strategies and improve vaccine uptake rates.


## Model Used
BaggingClassifier with XGBoostClassifier as the base estimator
## Key Points
* Addressed class imbalance
* Two separate models are instantiated using  BaggingClassifier with XGBoostClassifier as the base estimator for each target variable.
* Implemented hyperparameter tuning using Stratified RandomizedSearchCV and Stratified GridSearchCV.
* The best hyperparameters are directly used in the [code](https://github.com/Dream-Falls/AnalyticaX/blob/main/Source_code.py).
* Utilizes ROC AUC for evaluation within each model.
  Additionally, we calculate an overall ROC AUC by    combining both models' predictions, providing a comprehensive assessment of performance.
## Results
The predicted probabilities for the two target variables of the 'test_set_features' dataset are stored [here](https://github.com/Dream-Falls/AnalyticaX/blob/main/results.csv).

## Report

This [report](https://github.com/Dream-Falls/AnalyticaX/blob/main/report.pdf) demonstrates predictive models for H1N1 and seasonal flu vaccination probabilities, covering exploratory data analysis, preprocessing, model initialization, hyperparameter tuning, and evaluation metrics for informed public health strategies.



#  Predicting Probabilities of  H1N1 and Seasonal Vaccination Uptake

This project tackled the challenge of predicting the probabilities of individuals receiving H1N1 and seasonal influenza vaccinations using machine learning techniques. Leveraging data from the 2009 National H1N1 Flu Survey (NHFS) conducted by CDC, this project aimed to inform public health strategies and improve vaccine uptake rates. Notably, it achieved recognition by winning **2nd place** at **AnalyticaX**, a machine learning competition organized by IIT Indore during Fluxus'24.



## Model Used
BaggingClassifier with XGBoostClassifier as the base estimator
## Key Points
* Tackled the feature columns' missing values.
* Addressed the taget variables' class imbalance. 
* Two separate models are instantiated using BaggingClassifier with XGBoostClassifier as the base estimator for each target variable.
* Implemented hyperparameter tuning using Stratified RandomizedSearchCV and Stratified GridSearchCV.
* The best hyperparameters are directly used in the [code](https://github.com/Dream-Falls/AnalyticaX/blob/main/source_code.ipynb).
* Utilizes ROC AUC for evaluation within each model. Additionally, we calculated an overall ROC AUC by combining both models' predictions, providing a comprehensive assessment of performance.
## Results
The mean ROC AUC scores for each target variable and the overall score are summarized below:
   *  H1N1 vaccine: Mean ROC AUC score = 0.8415
   * Seasonal vaccine: Mean ROC AUC score = 0.8588
   * Overall ROC AUC score = 0.8501

     ![g14](https://github.com/Dream-Falls/AnalyticaX/assets/160475509/cd83bc46-19ec-4317-971d-96d2bd021567)


The predicted probabilities for the two target variables of the 'test_set_features' dataset are stored [here](https://github.com/Dream-Falls/AnalyticaX/blob/main/results.csv).

## Report

This [report](https://github.com/Dream-Falls/AnalyticaX/blob/main/report.pdf) demonstrates predictive models for H1N1 and seasonal flu vaccination probabilities, covering exploratory data analysis, preprocessing, model initialization, hyperparameter tuning, and evaluation metrics for informed public health strategies.

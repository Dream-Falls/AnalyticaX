

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
The mean ROC AUC scores for each target variable and the overall score are summarized below:
   *  H1N1 vaccine: Mean ROC AUC score = 0.8415
   * Seasonal vaccine: Mean ROC AUC score = 0.8588
   * Overall ROC AUC score = 0.8501

   
          ![g14](https://github-production-user-asset-6210df.s3.amazonaws.com/160475509/308976810-4ad93b4d-1aeb-456b-a1a9-891be64d16e7.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240229T152422Z&X-Amz-Expires=300&X-Amz-Signature=2a34c3df66e2808c11846538d8620ddfa3c5381401db39385f17318a26838eb0&X-Amz-SignedHeaders=host&actor_id=160475509&key_id=0&repo_id=759745250)


The predicted probabilities for the two target variables of the 'test_set_features' dataset are stored [here](https://github.com/Dream-Falls/AnalyticaX/blob/main/results.csv).

## Report

This [report](https://github.com/Dream-Falls/AnalyticaX/blob/main/report.pdf) demonstrates predictive models for H1N1 and seasonal flu vaccination probabilities, covering exploratory data analysis, preprocessing, model initialization, hyperparameter tuning, and evaluation metrics for informed public health strategies.

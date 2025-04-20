#overview
This project predicts the recurrence of thyroid cancer based on clinical features, using an XGBoost classifier.

#Dataset
Source: https://www.kaggle.com/datasets/aneevinay/thyroid-cancer-recurrence-dataset
The dataset consists of clinical records from 383 patients with thyroid cancer.
Features included: age, gender, treatment history, tumor characteristics, and metastasis information.
Target variable: "Recurred"

#Approach
Data Preprocessing: categorical features were label-encoded, minor typos in feature names were corrected.
Data Splitting: the data was stratified into Train (60%), Development/Validation (20%), and Test (20%) sets.
Model Building: XGBoost Classifier was used, initial hyperparameters were found through grid search.
Hyperparameter Tuning: manual tuning of max_depth, min_child_weight, and learning_rate based on
training and validation (development set) accuracies; visualized how model complexity affected performance.
Model Evaluation: Accuracy, precision, recall, and F1 score were calculated. Evaluation was performed separately on the development and test sets.

#Final model performance
Development Set Accuracy: ~96%
Test Set Accuracy: ~96%
Other Metrics: High precision and recall indicating balanced performance.

#Requirements
Python 3.x
pandas
numpy
scikit-learn
xgboost
matplotlib

#Usage
Run the main notebook or script to:
Preprocess the data
Train the model
Evaluate model performance
Visualize hyperparameter effects
Load the final trained model

#Highlights
Emphasis on model generalization (avoiding overfitting)
Careful hyperparameter tuning informed by visual analysis
Full separation of Train/Dev/Test datasets
Clean and reproducible code

#Future work
Test alternative models (Random Forest, LightGBM) for comparison.




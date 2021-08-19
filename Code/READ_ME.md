# Dataset:
The StudentLife dataset needs to be downloaded from the
StudentLife website.

Dataset: https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2

# General Info:
Two types of files exist: .py files which contain the code used to preprocess the data and .ipynb, which are jupyter notebooks where the models were trained and the results obtained.

# Preprocessing:
Sleep_stress_new_features.py -> In this file we extract the necessary features from the several data types of the StudentLife dataset (used in Stress_prediction_include_sms_call_log.ipynb and Sleep_prediction_include_sms_call_log.ipynb);

Sleep_stress_exclude_sms_call.py -> The only difference to the previous file is that we do not include the SMS and Call log data when creating each participant's dataframe (used in Stress_prediction_exclude_sms_call_log.ipynb and Sleep_prediction_exclude_sms_call_log.ipynb);

PHQ9_preprocessing.py -> Extracts the necessary features from the several data types to be used during the multivariate time series classification task (used in PHQ9_prediction.ipynb)

# Models and results:

PHQ9_prediction.ipynb -> The Fully Convolutional Network is built and trained to predict signs of depression in students;

Stress_prediction_exclude_sms_call_log.ipynb -> The XGBoost and TabNet models are built and trained to predict signs of stress in students, using the dataset that excludes the SMS and Call Log data;

Stress_prediction_include_sms_call_log.ipynb -> The XGBoost and TabNet models are built and trained to predict signs of stress in students, using the dataset that includes the SMS and Call Log data;

Sleep_prediction_exclude_sms_call_log.ipynb -> The XGBoost and TabNet models are built and trained to predict if a student will sleep the recommended amount of hours, using the dataset that excludes the SMS and Call Log data;

Sleep_prediction_include_sms_call_clog.ipynb -> The XGBoost and TabNet models are built and trained to predict if a student will sleep the recommended amount of hours, using the dataset that includes the SMS and Call Log data;

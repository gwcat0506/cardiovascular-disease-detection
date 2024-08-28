import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from itertools import combinations

# Start the timer
start_time = time.time()

# Load datasets
disease = "IHD" # HA, HF
japan = pd.read_csv("./JMDC_"+str(disease)+"_PATIENTS_top300.csv")
korea = pd.read_csv("./NHIS_"+str(disease)+"_PATIENTS_ver3.csv")

# Find common columns and filter datasets
common_columns = [col for col in japan.columns if col in korea.columns]
japan = japan[common_columns].drop(columns=['member_id']).fillna(0)
korea = korea[common_columns].drop(columns=['member_id']).dropna()

japan.reset_index(drop=True, inplace=True)
korea.reset_index(drop=True, inplace=True)

# Convert categorical columns to string type
categorical_columns = [disease,'icd10_level2_name','gender_of_member', 'smoking_habit', 'drinking_habit', 'exercise_habit', 'physical_activity']
japan[categorical_columns] = japan[categorical_columns].astype('str')
korea[categorical_columns] = korea[categorical_columns].astype('str')

# One-hot encoding for categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
columns_to_encode = ['gender_of_member', 'smoking_habit', 'drinking_habit', 'exercise_habit', 'physical_activity', 'icd10_level2_name']
encoded_japan = encoder.fit_transform(japan[columns_to_encode])
encoded_korea = encoder.transform(korea[columns_to_encode])

encoded_columns = encoder.get_feature_names_out(columns_to_encode)
encoded_japan_df = pd.DataFrame(encoded_japan, columns=encoded_columns)
encoded_korea_df = pd.DataFrame(encoded_korea, columns=encoded_columns)

japan = pd.concat([japan.drop(columns=columns_to_encode), encoded_japan_df], axis=1)
korea = pd.concat([korea.drop(columns=columns_to_encode), encoded_korea_df], axis=1)

# Standardization for numeric columns
features = ['bmi', 'systolic_bp', 'diastolic_bp', 'hdl_cholesterol', 'ldl_cholesterol', 'fasting_blood_sugar', 'AGE']
scaler_japan = StandardScaler()
japan[features] = scaler_japan.fit_transform(japan[features])

scaler_korea = StandardScaler()
korea[features] = scaler_korea.fit_transform(korea[features])

# Convert target column to int
japan[disease] = japan[disease].astype(int)
korea[disease] = korea[disease].astype(int)

# Define features and target
X = japan.drop(columns=[disease])
y = japan[disease]

# 1:1 Sampling(undersampling)
def perform_sampling(df, target_column):
    df_positive = df[df[target_column] == 1]
    df_negative = df[df[target_column] == 0].sample(len(df_positive), random_state=42)
    df_sampled = pd.concat([df_positive, df_negative])
    return df_sampled

korea = perform_sampling(korea, disease)
X_korea = korea.drop(columns=[disease])
y_korea = korea[disease]

# 5-fold setting
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_num = 1

# model init
svm_model = SVC(kernel='rbf', probability=True)
logreg_model = LogisticRegression(max_iter=100)
catboost_model = CatBoostClassifier(
                iterations=100,
                learning_rate=0.01,
                depth=6,
                loss_function='Logloss',
                verbose=False)

models = {
    'SVM': svm_model,
    'LogReg': logreg_model,
    'CatBoost': catboost_model
}

# all combinations setting
model_combinations = []
for i in range(1, len(models) + 1):
    model_combinations.extend(combinations(models.items(), i))

all_results = []
for combination in model_combinations:
    estimators = list(combination)
    ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
    
    japan_fold_results = []
    korea_fold_results = []
    
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the ensemble model
        ensemble_model.fit(X_train, y_train)

        # Predict on Japan test set
        y_pred_japan = ensemble_model.predict(X_test)
        
        # Predict on Korea test set 
        y_pred_korea = ensemble_model.predict(X_korea)
        
        # Classification report for Japan 
        report_dict_japan = classification_report(y_test, y_pred_japan, output_dict=True)
        
        # Classification report for Korea 
        report_dict_korea = classification_report(y_korea, y_pred_korea, output_dict=True)
        
        # save japan result
        j_fold_result = {
            'Model Combination': ' + '.join([name for name, _ in estimators]),
            'Fold': fold_num,
            'Accuracy': report_dict_japan['accuracy'],
            'Macro avg precision': report_dict_japan['macro avg']['precision'],
            'Macro avg recall': report_dict_japan['macro avg']['recall'],
            'Macro avg f1-score': report_dict_japan['macro avg']['f1-score'],
            'No disease precision': report_dict_japan['0']['precision'],
            'No disease recall': report_dict_japan['0']['recall'],
            'No disease f1-score': report_dict_japan['0']['f1-score'],
            'Yes disease precision': report_dict_japan['1']['precision'],
            'Yes disease recall': report_dict_japan['1']['recall'],
            'Yes disease f1-score': report_dict_japan['1']['f1-score']
        }
        
        # save korea result
        k_fold_result = {
            'Model Combination': ' + '.join([name for name, _ in estimators]),
            'Fold': fold_num,
            'Accuracy': report_dict_korea['accuracy'],
            'Macro avg precision': report_dict_korea['macro avg']['precision'],
            'Macro avg recall': report_dict_korea['macro avg']['recall'],
            'Macro avg f1-score': report_dict_korea['macro avg']['f1-score'],
            'No disease precision': report_dict_korea['0']['precision'],
            'No disease recall': report_dict_korea['0']['recall'],
            'No disease f1-score': report_dict_korea['0']['f1-score'],
            'Yes disease precision': report_dict_korea['1']['precision'],
            'Yes disease recall': report_dict_korea['1']['recall'],
            'Yes disease f1-score': report_dict_korea['1']['f1-score']
        }
        japan_fold_results.append(j_fold_result)
        korea_fold_results.append(k_fold_result)
        

    all_results.extend(japan_fold_results)
    all_results.extend(korea_fold_results)


results_df = pd.DataFrame(all_results)

# CSV save
results_df.to_csv(str(disease)+'_All_Ensemble_Results.csv', index=False)

# Finish time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")

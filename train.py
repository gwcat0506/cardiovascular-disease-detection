import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# Setting
disease = "IHD"  # "IHD", "HA", "HF", etc.
smote_apply = "True"  # Whether to apply SMOTE
undersampling = "False"  # Whether to apply undersampling
model_type = "LR"  # "LR", "RF", "SVM", "XGB", "CB", "TabNet"

# File paths
japan_file = f"./JMDC_{disease}_PATIENTS_top300.csv"
korea_file = f"./NHIS_{disease}_PATIENTS_ver3.csv"

# Data Load
japan = pd.read_csv(japan_file)
korea = pd.read_csv(korea_file)

# Common column extraction
common_columns = [col for col in japan.columns if col in korea.columns]
japan = japan[common_columns]
korea = korea[common_columns]

# Remove unnecessary columns and handle missing values
japan = japan.drop(columns=['member_id']).fillna(0)
korea = korea.drop(columns=['member_id']).dropna()

# Reset index
japan.reset_index(drop=True, inplace=True)
korea.reset_index(drop=True, inplace=True)

# Handle categorical columns
categorical_columns = [disease, 'icd10_level2_name', 'gender_of_member', 'smoking_habit', 'drinking_habit', 'exercise_habit', 'physical_activity']
japan[categorical_columns] = japan[categorical_columns].astype('str')
korea[categorical_columns] = korea[categorical_columns].astype('str')

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
columns_to_encode = ['gender_of_member', 'smoking_habit', 'drinking_habit', 'exercise_habit', 'physical_activity', 'icd10_level2_name']

encoded_japan = encoder.fit_transform(japan[columns_to_encode])
encoded_korea = encoder.transform(korea[columns_to_encode])

encoded_columns = encoder.get_feature_names_out(columns_to_encode)

encoded_japan_df = pd.DataFrame(encoded_japan, columns=encoded_columns)
encoded_korea_df = pd.DataFrame(encoded_korea, columns=encoded_columns)

japan = japan.drop(columns=columns_to_encode)
korea = korea.drop(columns=columns_to_encode)

japan = pd.concat([japan, encoded_japan_df], axis=1)
korea = pd.concat([korea, encoded_korea_df], axis=1)

# Standardization
features = ['bmi', 'systolic_bp', 'diastolic_bp', 'hdl_cholesterol', 'ldl_cholesterol', 'fasting_blood_sugar', 'AGE']

scaler_japan = StandardScaler()
japan[features] = scaler_japan.fit_transform(japan[features])

scaler_korea = StandardScaler()
korea[features] = scaler_korea.fit_transform(korea[features])

# Convert target variable
japan[disease] = japan[disease].astype(int)
korea[disease] = korea[disease].astype(int)

# Define 1:1 sampling function
def perform_sampling(df, target_column):
    df_positive = df[df[target_column] == 1]
    df_negative = df[df[target_column] == 0].sample(len(df_positive), random_state=42)
    df_sampled = pd.concat([df_positive, df_negative])
    return df_sampled

# Apply sampling
if undersampling:
    japan = perform_sampling(japan, disease)
korea = perform_sampling(korea, disease)

# Specify feature variables and target variable
X = japan.drop(columns=[disease])
y = japan[disease]

X_korea = korea.drop(columns=[disease])
y_korea = korea[disease]

# Cross-validation and model training
kf = StratifiedKFold(n_splits=5)
fold_num = 1
japan_fold_results = []
korea_fold_results = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    if smote_apply:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Model selection based on model_type variable
    if model_type == "LR":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "RF":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "SVM":
        model = SVC(kernel='rbf', probability=True)
    elif model_type == "XGB":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif model_type == "CB":
        model = CatBoostClassifier(iterations=100, learning_rate=0.01, depth=6, loss_function='Logloss', verbose=False)
    elif model_type == "TabNet":
        model = TabNetClassifier()
        X_train = X_train.values
        X_test = X_test.values
        X_korea = X_korea.values
    else:
        raise ValueError("Unsupported model type. Choose from 'LR', 'RF', 'SVM', 'XGB', 'CB', 'TabNet'.")

    # Train the model
    if model_type == "TabNet":
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_name=['valid'],
            max_epochs=50, patience=5,
            batch_size=256, virtual_batch_size=128,
            num_workers=0, drop_last=False)
        
    else:
        model.fit(X_train, y_train)
        
    y_pred_japan = model.predict(X_test)
    y_pred_korea = model.predict(X_korea)

    # Generate classification reports
    report_dict_japan = classification_report(y_test, y_pred_japan, output_dict=True)
    report_dict_korea = classification_report(y_korea, y_pred_korea, output_dict=True)

    j_fold_result = {
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

    k_fold_result = {
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

    fold_num += 1

# Save results to DataFrame
japan_results = pd.DataFrame(japan_fold_results)
korea_results = pd.DataFrame(korea_fold_results)

# Save results to CSV
japan_results.to_csv(f'{model_type}_japan_{disease}_result.csv', index=False)
korea_results.to_csv(f'{model_type}_korea_{disease}_result.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

disease = "IHD" # HA, HF

# Load data
patient_df = pd.read_csv(f'JMDC_{disease}_PATIENTS.csv')

# Data preprocessing
drop_cols = [col for col in patient_df.columns if len(col) == 3 and col.startswith('C')]
patient_df = patient_df.drop(columns=drop_cols)

# Split data into numerical and categorical features
X1 = patient_df.drop(columns=[disease, 'member_id', 'date_of_health_checkup', 'gender_of_member', 'icd10_level2_name', 'smoking_habit', 'drinking_habit', 'exercise_habit', 'physical_activity'])
X2 = patient_df[['gender_of_member', 'icd10_level2_name', 'smoking_habit', 'drinking_habit', 'exercise_habit', 'physical_activity']]

# Encode categorical variables
label_encoders = {
    'gender_of_member': LabelEncoder(),
    'smoking_habit': LabelEncoder(),
    'drinking_habit': LabelEncoder(),
    'exercise_habit': LabelEncoder(),
    'physical_activity': LabelEncoder(),
    'icd10_level2_name': LabelEncoder()
}

for col, encoder in label_encoders.items():
    X2[col] = encoder.fit_transform(X2[col])

# Combine numerical and categorical features
X = pd.concat([X1, X2], axis=1)
X.fillna(0, inplace=True)

# Labels
y = patient_df[disease]

# Create DMatrix for XGBoost
inputs = xgb.DMatrix(X, label=y)

# XGBoost parameters
params = {
    'max_depth': 6,
    'eta': 0.3,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Train the model
num_round = 100
bst = xgb.train(params, inputs, num_round)

# Plot feature importance
ax = xgb.plot_importance(bst, max_num_features=10, height=0.2, title=f"{disease} Feature Importance", ylabel='')
plt.subplots_adjust(left=0.2)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
plt.savefig(f'Best_Features_{disease}_JMDC.png')

# Get and save top 300 features
importance_scores = bst.get_score(importance_type='weight')
sorted_importance = sorted(importance_scores.items(), key=lambda item: item[1], reverse=True)

# List of necessary columns
necessary_cols = ['member_id', 'gender_of_member', 'date_of_health_checkup', 'bmi', 'systolic_bp', 
                  'diastolic_bp', 'hdl_cholesterol', 'ldl_cholesterol', 'fasting_blood_sugar', 
                  'smoking_habit', 'drinking_habit', 'exercise_habit', 'physical_activity', 'AGE', 
                  'icd10_level2_name']

# Filter out necessary columns from the sorted importance list
sorted_importance_fin = [item for item in sorted_importance if item[0] not in necessary_cols]
top_features = sorted_importance_fin[:300]

# Collect final set of features
feature_use = [col for col, _ in top_features]
feature_use.extend(necessary_cols)
feature_use.append(disease)

# Save the final dataset with top 300 features
patient_df[feature_use].to_csv(f'JMDC_{disease}_PATIENTS_top300.csv', index=False)

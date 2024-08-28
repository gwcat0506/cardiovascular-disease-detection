import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Define 1:1 Undersampling
def perform_sampling(df, target_column):
    df_positive = df[df[target_column] == 1]
    df_negative = df[df_positive.columns].sample(len(df_positive), random_state=42)
    df_sampled = pd.concat([df_positive, df_negative])
    return df_sampled

# Specify the disease type
disease = "IHD"  # HA, HF
smote_apply = "True"  # Whether to apply SMOTE
undersampling = "False"  # Whether to apply undersampling

# File paths
japan_file = f"./JMDC_{disease}_PATIENTS_top300.csv"
korea_file = f"./NHIS_{disease}_PATIENTS_ver3.csv"

# Data Load
japan = pd.read_csv(japan_file)
korea = pd.read_csv(korea_file)

# Extract common columns
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

# Convert target variable to integer
japan[disease] = japan[disease].astype(int)
korea[disease] = korea[disease].astype(int)

# Apply sampling if required
if undersampling:
    japan = perform_sampling(japan, disease)
korea = perform_sampling(korea, disease)

# Prepare numpy arrays for PyTorch
X_np = japan.drop(columns=[disease]).values
y_np = japan[disease].values
X_korea = korea.drop(columns=[disease]).values
y_korea = korea[disease].values

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
num_epochs = 50
patience = 3
batch_size = 64
learning_rate = 0.001

# Initialize Stratified K-Fold (5-Fold)
kf = StratifiedKFold(n_splits=5)
fold_results_japan = []
fold_results_korea = []
fold_num = 1

for train_valid_index, test_index in kf.split(X_np, y_np):
    # Split Train+Validation set and Test set (8:2 ratio)
    X_train_valid, X_test_fold = X_np[train_valid_index], X_np[test_index]
    y_train_valid, y_test_fold = y_np[train_valid_index], y_np[test_index]

    # Split Train+Validation into Train and Validation sets (75% Train, 25% Validation)
    X_train_fold, X_valid_fold, y_train_fold, y_valid_fold = train_test_split(
        X_train_valid, y_train_valid, test_size=0.25, random_state=fold_num
    )

    # Prepare TensorDataset and DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32), torch.tensor(y_train_fold, dtype=torch.float32))
    valid_dataset = TensorDataset(torch.tensor(X_valid_fold, dtype=torch.float32), torch.tensor(y_valid_fold, dtype=torch.float32))
    test_japan_dataset = TensorDataset(torch.tensor(X_test_fold, dtype=torch.float32), torch.tensor(y_test_fold, dtype=torch.float32))
    test_korea_dataset = TensorDataset(torch.tensor(X_korea, dtype=torch.float32), torch.tensor(y_korea, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_japan_loader = DataLoader(test_japan_dataset, batch_size=batch_size, shuffle=False)
    test_korea_loader = DataLoader(test_korea_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = X_np.shape[1]
    model = MLP(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set up Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Train model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validate model
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item() * inputs.size(0)

        valid_loss /= len(valid_loader.dataset)

        # Check Early Stopping
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            patience_counter = 0
            best_model_wts = model.state_dict()  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Load best model
    model.load_state_dict(best_model_wts)

    # Predict on Japanese data
    model.eval()
    y_pred_japan = []
    with torch.no_grad():
        for inputs, _ in test_japan_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_pred_japan.extend(outputs.cpu().numpy())

    y_pred_japan = np.array(y_pred_japan).round()

    # Predict on Korean data
    y_pred_korea = []
    with torch.no_grad():
        for inputs, _ in test_korea_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_pred_korea.extend(outputs.cpu().numpy())

    y_pred_korea = np.array(y_pred_korea).round()

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
japan_results.to_csv(f'MLP_japan_{disease}_result.csv', index=False)
korea_results.to_csv(f'MLP_korea_{disease}_result.csv', index=False)

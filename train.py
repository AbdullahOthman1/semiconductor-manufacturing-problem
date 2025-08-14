import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

import joblib


def drop_highly_correlated(df, threshold=0.95):
    X = df.drop(columns='Pass/Fail')
    y = df['Pass/Fail']
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Dropping {len(to_drop)} highly correlated features (correlation > {threshold})")
    reduced_df = X.drop(columns=to_drop)
    reduced_df['Pass/Fail'] = y
    return reduced_df


splits = {}
def split_data(df, name, test_size=0.2, random_state=42):
    X = df.drop(columns='Pass/Fail').values
    y = df['Pass/Fail'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    splits[name] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    print(f"[{name}]: Train: {X_train.shape}, Test: {X_test.shape}, Class balance (train): {np.bincount(y_train)}, (test): {np.bincount(y_test)}")




parquet_file = "data/uci-secom.parquet"
df = pd.read_parquet(parquet_file, engine='pyarrow')
df.head()

X = df.drop(columns=['Pass/Fail', 'Time'], errors='ignore')
y = df['Pass/Fail'].values

window_size = 11
X_rolled = X.copy()

for col in X_rolled.columns:
    if X_rolled[col].isna().any():
        X_rolled[col] = X_rolled[col].fillna(
            X_rolled[col].rolling(window=window_size, center=True, min_periods=1).median()
        )


X_rolled = X_rolled.fillna(X_rolled.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_rolled)


rf_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
rf_model.fit(X_scaled, y)

rf_importances = rf_model.feature_importances_


rf_importances_df = pd.DataFrame({
    'feature': X.columns,
    'rf_importance': rf_importances
}).sort_values(by='rf_importance', ascending=False)

threshold_rf = 0.90
rf_cum_importance = rf_importances_df['rf_importance'].cumsum()
rf_selected_features = rf_importances_df[rf_cum_importance <= threshold_rf]['feature'].tolist()

# Create reduced feature DataFrame for RandomForest
X_rf = X_scaled[:, [X.columns.get_loc(feat) for feat in rf_selected_features]]
rf_df = pd.DataFrame(X_rf, columns=rf_selected_features)
rf_df['Pass/Fail'] = y

rf_df_uncorr = drop_highly_correlated(rf_df, threshold=0.9)

split_data(rf_df_uncorr, "RF_uncorr")


model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=12,
    random_state=42
)

data = splits["RF_uncorr"]
X, y = data['X_train'], data['y_train']

cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

scores = cross_validate(model, X, y, cv=cv, scoring=['f1','roc_auc'])

print("CV Scores:", scores)

model.fit(X, y)

joblib.dump(model, "models/xgb_model.pkl")
print("Model saved to xgb_model.pkl")

print(f"F1-score mean: {scores['test_f1'].mean()}")
print(f"AUC-score mean: {scores['test_roc_auc'].mean()}")
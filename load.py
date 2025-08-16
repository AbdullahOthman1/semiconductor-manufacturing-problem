import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

model = joblib.load("models/xgb_model.pkl")
data= pd.read_parquet("data/uncorrelated_data.parquet")

y= data['Pass/Fail']
X= data.drop(columns='Pass/Fail')


X_train, X_test, y_train, y_test= train_test_split(X, y,
        test_size=0.2,
        random_state=50,
        stratify=y)

print(model.predict(X_test))

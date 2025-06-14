from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import numpy as np
import pandas as pd
# Features and targe
# import pandas as pd
df = pd.read_csv('latestnewdataset.csv')
target = 'Price (Lakhs)'
features = ['Area', 'Area (sq.ft.)', 'BHK', 'Bathrooms', 'Furnishing Status',
            'Distance to School (km)', 'Distance to Hospital (km)',
            'Distance to Metro (km)', 'Age of Property (years)']

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing
categorical = ['Area', 'Furnishing Status']
numerical = ['Area (sq.ft.)', 'BHK', 'Bathrooms', 'Distance to School (km)',
             'Distance to Hospital (km)', 'Distance to Metro (km)', 'Age of Property (years)']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numerical)
])

# Pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Save model and preprocessor
joblib.dump(pipeline, 'house_price_model.pkl')
print(rmse)
print(r2)
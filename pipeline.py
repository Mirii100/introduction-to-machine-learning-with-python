# creating  a pipeline   
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer # aalows to link pipeline together 
from sklearn.preprocessing import StandardScaler, OneHotEncoder # for numerical and categorical features
from sklearn.impute import SimpleImputer # for handling missing values
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# load data

data=pd.read_csv('myvehicle.csv')
data.info()
data.head()
data.isnull().sum()

# creating features and target
x= data.drop(["CO2_Emissions"],axis=1)
y=data["CO2_Emissions"]

# split data
numerical_cols=["Model_Year","Engine_Size","Cylinders","Fuel_Consumption_in_City(L/100 km)","Fuel_Consumption_in_City_Hwy(L/100 km)","Fuel_Consumption_comb(L/100km)","Smog_Level"]
categorical_cols=["Make","Model","Vehicle_Class","Transmission"]

# starting pipeline 
numerical_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='mean')), # drops a column if it is empty 
    ('scaler',StandardScaler()) # standardizes the data
])

categorical_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')), # fills with most frequent value 
    ('encoder',OneHotEncoder(handle_unknown='ignore')) # creates a new column for each unique value
])
# joining pipeine together

preprocessor=ColumnTransformer([
    ('num',numerical_pipeline,numerical_cols),# should accept numerical columns
    ('cat',categorical_pipeline,categorical_cols) # should accept categorical columns
])
pipeline=Pipeline([
    ('preprocessor',preprocessor),
    ('model',RandomForestRegressor())
])

# splitting data
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# training model
pipeline.fit(X_train,y_train)

# prediction
prediction=pipeline.predict(X_test)

# view the encoding
encoded_cols=pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols)
print(encoded_cols)

# evaluate 
mse=mean_squared_error(y_test,prediction) # average squared error
rmse=np.sqrt(mse) # root mean squared error

r2=r2_score(y_test,prediction) # coefficient of determination
mae=mean_absolute_error(y_test,prediction) # mean absolute error

print(f'Mean Squared Error :{mse:.2f}')
print(f'Root Mean Squared Error :{rmse:.2f}')
print(f'R2 Score :{r2:.2f}')
print(f'Mean Absolute Error :{mae:.2f}')

# saving a pipeline 
joblib.dump(pipeline,'vehicle_pipeline.joblib') # save the pipeline 

plt.figure(figsize=(8, 6))
plt.scatter(y_test, prediction, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')  # Perfect prediction line
plt.xlabel("Actual CO₂ Emissions")
plt.ylabel("Predicted CO₂ Emissions")
plt.title("Actual vs. Predicted CO₂ Emissions")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual CO₂ Emissions", marker='o')
plt.plot(prediction, label="Predicted CO₂ Emissions", marker='x')
plt.xlabel("Test Samples")
plt.ylabel("CO₂ Emissions")
plt.title("Comparison of Actual and Predicted CO₂ Emissions")
plt.legend()
plt.grid(True)
plt.show()


importances = pipeline.named_steps['model'].feature_importances_
feature_names = numerical_cols + list(encoded_cols)  # Combining numerical and encoded categorical features

# Sorting the feature importance
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_indices], y=np.array(feature_names)[sorted_indices], palette="viridis")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in CO₂ Emission Prediction")
plt.grid(True)
plt.show()

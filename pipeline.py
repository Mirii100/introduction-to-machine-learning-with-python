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

data=pd.read_csv('vehicle.csv')
data.info()

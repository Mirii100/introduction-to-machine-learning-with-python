import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV


 # help split data for training  for a model

from sklearn.preprocessing import MinMaxScaler  # convert into a range of zero to one 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics  import accuracy_score ,confusion_matrix # tracks our performance 


# for visualization 
import matplotlib.pyplot as plt 
import seaborn as sns 



# creating a dataframe 

data = pd.read_csv('mytitanic.csv')
data.info()# getting information on the data 
print(data.isnull().sum())# get null values 
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


# data cleaning 

def preprocess_data(df):
    df.drop(columns=["PassengerId","Name","Ticket","Cabin" ], inplace=True)
    df["Embarked"].fillna("S",inplace=True )# fill missing data with s
    df.drop(columns=["Embarked"],inplace=True)

    fill_missing_ages(df)

    # convert gender 
    df["Sex"]=df["Sex"].map({'male':1,'female':0})

    # feature engineering
    df["FamilySize"]=df["SibSp"]+df["Parch"]
    df["IsAlone"]=np.where(df["FamilySize"]==0, 1,0)
    df["FareBin"]=pd.qcut(df["Fare"],4,labels=False)
    df["AgeBin"]=pd.cut(df["Age"],bins=[0,12,20,40,60,np.inf],labels=False)

    return df

# fill in missing ages
def fill_missing_ages(df):
    age_fill_map={}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass]=df[df["Pclass"]==pclass]["Age"].median()

    df["Age"]=df.apply(lambda row :age_fill_map[row["Pclass"] ]  if pd.isnull(row["Age"]) else row["Age"] ,axis=1) # allows us to do a custom function  amnd check whether the age is missing 


# giving original dataframe 
data=preprocess_data(data)


# creating features /targets variables

x=data.drop(columns=["Survived"])
y=data["Survived"]

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42) # splits the data into 4 


# machine learning preprocessing 

scaler=MinMaxScaler() # creating a scaler 

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# hyper parameter tuning -knn

def tune_model(X_train,y_train):
    param_grid={
        "n_neighbors":range(1,2),
        "metric":["euclidean","manhattan","minkowski"],
        "weights":["uniform","distance"]
    }
# model
    model=KNeighborsClassifier()
    grid_search=GridSearchCV(model,param_grid,cv=5,n_jobs=-1)
    grid_search.fit(X_train,y_train)# allows us to train data 
    
    return grid_search.best_estimator_


best_model=tune_model(X_train,y_train) # this is the model 


# prediction

def evaluate_model(model,X_test,y_test):
    prediction=model.predict(X_test)
    accuracy=accuracy_score(y_test,prediction) # give the prediction and solution 
    matrix=confusion_matrix(y_test,prediction)
    return accuracy,matrix


accuracy,matrix=evaluate_model(best_model,X_test,y_test)#calling the function 

print(f'Accuracy is :{accuracy*100:.2f}%')
print('confusion matrix :')
print(matrix)


# plotting 
def plot_model(matrix):
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix,annot=True,fmt="d",xticklabels=["Survived","Not survived "],yticklabels=["Not surived","Survived"])# shows confusin matrix very nice 
    plt.title("confusion matrix ")
    plt.xlabel("predicted values ")
    plt.ylabel("actual values ")
    plt.show()


plot_model(matrix)





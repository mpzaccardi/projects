'''
Michael Zaccardi
IDSN 599, Spring 2021
zaccardi@usc.edu
Final Project Part 3
'''

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import os
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


DIABETES_PATH = os.path.join("datasets", "diabetes")

#Open up the file housing.csv and convert the data into a pandas DataFrame object
def load_diabetes_data(diabetes_path = DIABETES_PATH):
    csv_path = os.path.join(diabetes_path, "diabetes.csv")
    return pd.read_csv(csv_path)

diabetes = load_diabetes_data()

# split data set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(diabetes, diabetes["Diabetes_012"]):
    strat_train_set = diabetes.loc[train_index]
    strat_test_set = diabetes.loc[test_index]


dropCat = ['Diabetes_012', 'Veggies', 'HvyAlcoholConsump', 'Fruits', 'AnyHealthcare', 'Sex', 'NoDocbcCost']

diabetes = strat_train_set.copy()
diabetes = strat_train_set.drop(dropCat, axis=1)
diabetes_diagnosis = strat_train_set["Diabetes_012"].copy()

diabetes_test = strat_test_set.drop(dropCat, axis=1)
diabetes_diagnosis_test = strat_test_set["Diabetes_012"].copy()


# data info
'''
diabetes_matrix = diabetes.corr()
print(diabetes_matrix['Diabetes_012'].sort_values(ascending=False))
'''
diabetes.info()
'''
print(diabetes["Income"].value_counts()) 
print(diabetes.describe()) #This method shows a summary of the numerical attributes
diabetes.hist(bins=50, figsize=(20,15)) #shows the number of instances (vertical axis) that have a given value range
#plt.show()#Plots a histogram for each numerical attribute


Attributes = ['GenHlth', 'HighBP', 'BMI', 'DiffWalk']
scatter_matrix(diabetes[Attributes], figsize=(12, 8))
plt.show()
'''

#Transformers

#scaler = StandardScaler()
scaler = MinMaxScaler()
numeric_features = ['BMI', 'MentHlth', 'PhysHlth']
#numeric_features = []

transformer = ColumnTransformer([('num', scaler, numeric_features)], remainder='passthrough')

diabetes_prepared = transformer.fit_transform(diabetes)
print(diabetes_prepared)


# Random Forest

param_grid = [
    {'bootstrap': [True], 'n_estimators': [30], 'max_features': [10]},]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                            scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(diabetes_prepared, diabetes_diagnosis)

print(grid_search.best_params_)

print(grid_search.best_estimator_)

cvres = grid_search.cv_results_



#Random Performance


for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

some_data = diabetes_test.iloc[:100]
some_labels = diabetes_diagnosis_test.iloc[:100]
some_data_prepared = transformer.transform(some_data)

predictions = []
for x in grid_search.predict(some_data_prepared):
    if x >= 1:
        x = 2
    else:
        x = 0
    predictions.append(x)

print("Predictions:", predictions)
print("Labels:", list(some_labels))

some_data = diabetes_test.iloc[:100]
some_labels = diabetes_diagnosis_test.iloc[:100]
some_data_prepared = transformer.transform(some_data)
some_data_predicted = grid_search.predict(some_data_prepared)
print("Predictions:", grid_search.predict(some_data_prepared))
print("Labels:", list(some_labels))

tree_mse = mean_squared_error(some_labels, some_data_predicted)
tree_rmse = np.sqrt(tree_mse)

print(tree_mse)
print(tree_rmse)
'''

# Linear Model



linear_model = LogisticRegression(max_iter=10000)
diabetes_linear = linear_model.fit(diabetes_prepared, diabetes_diagnosis)


some_data = diabetes.iloc[:100]
some_labels = diabetes_diagnosis.iloc[:100]
some_data_prepared = transformer.transform(some_data)
print("Predictions:", linear_model.predict(some_data_prepared))
print("Labels:", list(some_labels))


#Linear Performance


diabetes_predictions = linear_model.predict(diabetes_prepared)
lin_mse = mean_squared_error(diabetes_diagnosis, diabetes_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

some_data = diabetes_test
some_labels = diabetes_diagnosis_test
some_data_prepared = transformer.transform(some_data)
some_data_predicted = linear_model.predict(some_data_prepared)
print("Predictions:", linear_model.predict(some_data_prepared))
print("Labels:", list(some_labels))

tree_mse = mean_squared_error(some_labels, some_data_predicted)
tree_rmse = np.sqrt(tree_mse)

print(tree_mse)
print(tree_rmse)


#Decision Tree


tree_reg = DecisionTreeRegressor()
tree_reg.fit(diabetes_prepared, diabetes_diagnosis)
diabetes_predictions = tree_reg.predict(diabetes_prepared)
tree_mse = mean_squared_error(diabetes_diagnosis, diabetes_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

#Decision Tree Performance

some_data = diabetes_test
some_labels = diabetes_diagnosis_test
some_data_prepared = transformer.transform(some_data)
some_data_predicted = tree_reg.predict(some_data_prepared)
print("Predictions:", tree_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

tree_mse = mean_squared_error(some_labels, some_data_predicted)
tree_rmse = np.sqrt(tree_mse)

print(tree_mse)
print(tree_rmse)
'''


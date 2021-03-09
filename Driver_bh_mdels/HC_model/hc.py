import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
# from sklearn.utils import shuffle
# import matplotlib.pyplot as pyplot
import pickle
# from matplotlib import style

# Loading the data
path = 'C:/Users/phillemon/Documents/Work/Models/Harsh conering/data/train0.csv'
data = pd.read_csv(path, delimiter=",",header=None, skipinitialspace=True)
data.columns=['Y', 'X', 'Z', 'X_Contr', 'Target']

# Extracting fields to work with
data = data[['Y', 'X', 'Z', 'Target']]
# The value to predict from the data
predict = "Target"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Splitting about 10% of the data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)
best = 0
for _ in range(30):
    # Splitting about 10% of the data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    # saving the model
    if acc > best:
        best = acc
        with open("Harsh_Con_model.pickle", "wb") as f:
            pickle.dump(linear, f)

# Loading the saved model
pickle_in = open("Harsh_Con_model.pickle", "rb")
model = pickle.load(pickle_in)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])



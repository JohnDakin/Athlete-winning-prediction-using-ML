import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


teams = pd.read_csv("teams.csv")

teams

teams = teams[["team","country","year","athletes","age","prev_medals","medals"]]

teams

# teams.corr()["medals"]
teams.select_dtypes(include=['number']).corr()['medals']

sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)

sns.lmplot(x="age", y="medals", data=teams, fit_reg=True, ci=None)

teams.plot.hist(y="medals")


#Cleaning data

teams[teams.isnull().any(axis=1)]

teams = teams.dropna()

teams

#Split data
#Train data and Test sat
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

#Rows in training set
train.shape

#Rows in testing set
test.shape

#Training using mean absolute metric
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
target = "medals"

reg.fit(train[predictors], train["medals"])

LinearRegression()

predictions = reg.predict(test[predictors])

# print(predictions)
test["predictions"] = predictions
test
test.loc[test["predictions"] < 0, "predictions"]  = 0
test["predictions"] = test["predictions"].round()

test

#mean absolute error
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(test["medals"], test["predictions"])

error

teams.describe()["medals"]

test[test["team"] == "USA"]
test[test["team"] == "IND"]

errors = (test["medals"] - test["predictions"]).abs()

errors

error_by_team = errors.groupby(test["team"]).mean()

error_by_team

medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio = error_by_team / medals_by_team
error_ratio

error_ratio[~pd.isnull(error_ratio)]
import numpy as np
error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio

error_ratio.plot.hist()
plt.show()

error_ratio.sort_values()

#Add in more predictors
#Try different models e.g. random forest or nural network

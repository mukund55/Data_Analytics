# Waiter Tips Prediction using Python

import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

data = pd.read_csv("tips.csv")

print(data.head())

----------------------------------------------------------------------------------------------------------------------------------------
   total_bill   tip     sex smoker  day    time  size
   
0       16.99  1.01  Female     No  Sun  Dinner     2

1       10.34  1.66    Male     No  Sun  Dinner     3

2       21.01  3.50    Male     No  Sun  Dinner     3

3       23.68  3.31    Male     No  Sun  Dinner     2

4       24.59  3.61  Female     No  Sun  Dinner     4

----------------------------------------------------------------------------------------------------------------------------------------

Below is the complete description of this dataset:

    1. total_bill: Total bill in dollars including tax
    2. tip: Tip given to waiter in dollars
    3. sex: gender of the person paying the bill
    4. smoker: whether the person smoked or not
    5. day: day of the week
    6. time: lunch or dinner
    7. size: number of people

Now let’s move forward by analyzing all the factors affecting waiter tips.

----------------------------------------------------------------------------------------------------------------------------------------
Let’s have a look at the tips given to the waiters according to:
    1. the total bill paid
    2. number of people at a table
    3. and the day of the week:
----------------------------------------------------------------------------------------------------------------------------------------
  
figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "day", trendline="ols")
                    
figure.show()

----------------------------------------------------------------------------------------------------------------------------------------

Now let’s have a look at the tips given to the waiters according to: 
    1. the total bill paid
    2. the number of people at a table
    3. and the gender of the person paying the bill:

----------------------------------------------------------------------------------------------------------------------------------------

figure = px.scatter(data_frame = data, x="total_bill", y="tip", size="size", color= "sex", trendline="ols")

figure.show()

----------------------------------------------------------------------------------------------------------------------------------------

Now let’s have a look at the tips given to the waiters according to:
    1. the total bill paid
    2. the number of people at a table
    3. and the time of the meal:

----------------------------------------------------------------------------------------------------------------------------------------

figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "time", trendline="ols")
                    
figure.show()

----------------------------------------------------------------------------------------------------------------------------------------

Now let’s see the tips given to the waiters according to the days to find out which day the most tips are given to the waiters:

----------------------------------------------------------------------------------------------------------------------------------------

figure = px.pie(data, 
             values='tip', 
             names='day',hole = 0.5)
             
figure.show()

----------------------------------------------------------------------------------------------------------------------------------------

According to the visualization above, on Saturdays, most tips are given to the waiters.
Now let’s look at the number of tips given to waiters by gender of the person paying the bill to see who tips waiters the most:

----------------------------------------------------------------------------------------------------------------------------------------

figure = px.pie(data, 
             values='tip', 
             names='sex',hole = 0.5)
             
figure.show()

----------------------------------------------------------------------------------------------------------------------------------------

According to the visualization above, most tips are given by men. Now let’s see if a smoker tips more or a non-smoker:

----------------------------------------------------------------------------------------------------------------------------------------

figure = px.pie(data, 
             values='tip', 
             names='smoker',hole = 0.5)
             
figure.show()

----------------------------------------------------------------------------------------------------------------------------------------

According to the visualization above, non-smoker tips waiters more than smokers. Now let’s see if most tips are given during lunch or dinner:

----------------------------------------------------------------------------------------------------------------------------------------

figure = px.pie(data, 
             values='tip', 
             names='time',hole = 0.5)
             
figure.show()

----------------------------------------------------------------------------------------------------------------------------------------

According to the visualization above, a waiter is tipped more during dinner.
So this is how we can analyze all the factors affecting waiter tips. Now in the section below, I will take you through how to train a machine learning model for the task of waiter tips prediction.

Waiter Tips Prediction Model
Before training a waiter tips prediction model, I will do some data transformation by transforming the categorical values into numerical values:

----------------------------------------------------------------------------------------------------------------------------------------

data["sex"] = data["sex"].map({"Female": 0, "Male": 1})

data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})

data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})

data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})

data.head()

----------------------------------------------------------------------------------------------------------------------------------------

   total_bill   tip  sex  smoker  day  time  size
   
0       16.99  1.01    0       0    3     1     2

1       10.34  1.66    1       0    3     1     3

2       21.01  3.50    1       0    3     1     3

3       23.68  3.31    1       0    3     1     2

4       24.59  3.61    0       0    3     1     4

----------------------------------------------------------------------------------------------------------------------------------------

Now I will split the data into training and test sets:

----------------------------------------------------------------------------------------------------------------------------------------

x = np.array(data[["total_bill", "sex", "smoker", "day", 
                   "time", "size"]])
                   
y = np.array(data["tip"])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

----------------------------------------------------------------------------------------------------------------------------------------

Now below is how we can train a machine learning model for the task of waiter tips prediction using Python:

----------------------------------------------------------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(xtrain, ytrain)

----------------------------------------------------------------------------------------------------------------------------------------

Now let’s test the performance of this model by giving inputs to this model according to the features that we have used to train this model:

----------------------------------------------------------------------------------------------------------------------------------------

features = [[total_bill, "sex", "smoker", "day", "time", "size"]]

features = np.array([[24.50, 1, 0, 0, 1, 4]])

model.predict(features)

---
layout: post
title: Basic Machine Learning models
subtitle: Learning about Decission trees and random forests.
tags: [python, scikitlearn, kaggle]
---

So, I've been taking the avalaible courses in [Kaggle.com][kaggle website] for a while now. I started with their python course so I could review my knowledge of the language  
After refreshing myself about python, I continued with the Intro to machine learning course. Here are the basics I want to share:

The first thing they teach, is a model, what is a model?  
The most simple and basic model is called the decission tree, and it looks something like this:
![Decission tree image](https://i.imgur.com/R3ywQsR.png){: .mx-auto.d-block :}

To build a decission tree model with a set of data you have, let's see the following code:

{% highlight python linenos %}
# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))
{% endhighlight %}

In the code above, we use some external libraries such as pandas and sklearn, this libraries helps us to have a better data management within python.  
After we have the needed libraries, we specify the path of our file .csv wich contains our data.  
In the code we have some commentary that helps up understand what are we doing, for example, after specifying the data file path, we create a target object **y** wich is what we want to predict, and a set of features **X** we'll use to predict our y target.  
After that, we split our data into a set of training data and a set of validation data. We'll fit (or train) our model with some data and test it with different data so we can measure the accuracy of our model.  
Specify and fit the model is as easy as the code above says, the "random_state" allows us to get the same results every time we run it with a numeric value.

That's it to get a model that works, now we need to measure its accuracy.  
We first use our validation data to see what our model can predict, and after that we calculate the mean absolute error of those predictions vs validation data.  

What we want is a model that gives us the minimun mean absolute error possible, and we can change the attributes given to the model to approach that minimun.  
For example:

```python
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
    
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
maeOfEachMax = []
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for maxLeafNode in candidate_max_leaf_nodes:
    mae = get_mae(maxLeafNode, train_X, val_X, train_y, val_y)
    maeOfEachMax.append(mae)
    
# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
bestMae = min(maeOfEachMax)
maxLeafIndex = maeOfEachMax.index(bestMae)
best_tree_size = candidate_max_leaf_nodes[maxLeafIndex]

print(best_tree_size)
print(bestMae)

```

The code above iterates among the possible max_leaf_nodes value we can pass to our decission tree model, and gives us the minimun mean absolute error possible within this options.  
Something we can do with this information is fit the model with **all our data**, since we don't need the validation data to measure it, because we already "found the best model" we can build, although, we'll use those data in the following code:

```python
# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))
```

And that's our best decission tree model!  
We can still tune it in order to get better results with a lesser _mae_, but we can use other model instead:

A random forest model uses several decission trees's predictions and evaluates the mean of their mae to give a prediction. We can build one like this:

```python
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state = 1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(rf_model.predict(val_X), val_y)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
```

And as easy as that! Our validation MAE with a default random forest model is lower than our validation MAE with a tuned decission tree model.


This are the basics I got from the Intro to machine learning course of Kaggle, I'll now go with the intermediate course and see some contests in Kaggle about data analysis. I hope I can keep up with next steps.

Greetings! And thanks for reading.


[kaggle website]: https://www.kaggle.com/

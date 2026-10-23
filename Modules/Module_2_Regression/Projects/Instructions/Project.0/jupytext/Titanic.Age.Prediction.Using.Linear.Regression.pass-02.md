# Linear Regression to Predict Titanic Passenger Age


**Deep Dive Coding Data Science Bootcamp Sample Project**

---
Robert Citek - October 5th, 2026



## Problem Definition

State the business problem. Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.  Mention which model will be used, if known.


### Regression

For a regression, the Age will be the target.



## Data Collection/Sources


### Data Overview


This data comes from [Kaggle](https://www.kaggle.com/c/titanic/data), which was then copied to AWS for easy access.

https://ddc-datascience.s3.amazonaws.com/Projects/Example/Data/Titanic.train.csv



### Imports

```python
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn import datasets, metrics, model_selection
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```

### File Paths

Create a variable `url` that contains the web link to the CSV data

```python
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Example/Data/Titanic.train.csv'    # insert url here
url
```

### Load data

Load the data into a data frame named `titanic_df`.

```python
# load data
titanic_df = pd.read_csv( url )
```


### head


Look at the first few rows

```python
# first few rows
titanic_df.head()
```


### tail


Look at the last few rows


```python
# last few rows
titanic_df.tail()
```


### shape


```python
# shape of data frame
titanic_df.shape
```

### size


```python
# size of data frame
titanic_df.size
```


### Summary meta-data


Run the `info()` method on the data frame.


```python
# display meta-data
titanic_df.info()
```

### Summary statistics

Summary statistics give you a quick overview of the data.  Some items to spot check:
- the Target:
  - look for nulls
    - null will be culled
  - look at min, max, median, and mean
    - is it a categorical field masquerading as a number?
- the Features:
  - numerical
    - look at min, max, median, and mean
      - a min=1 and max=count suggests an ID
  - categorical
    - look at unique and freq
      - freq=1, unique=count suggests an ID
      - freq=count suggests a meaningless feature


Run the `describe()` method on the data frame.

```python
# display summary statistics
titanic_df.describe().transpose()
```


### target nulls



The target should be the first field that you look at.  Its data type will tell you what kind of machine learning problem this is ( classification or regression. ) Any nulls should be culled.  You can check for any skew in the data set.

```python
target_col = 'Age'
titanic_df[target_col]
```

```python
# look for nulls in the target
titanic_df[target_col].isnull().sum() / titanic_df[target_col].shape[0] * 100
# I am comfortable dropping nulls if 20% or under
```


### Nulls



```python
# look for nulls
titanic_df.drop(columns=[target_col]).isnull().sum().sort_values(ascending=False)
# Cabin and Embarked have nulls
```


```python
# look for nulls, displaying relative frequency
(titanic_df.drop(columns=[target_col]).isnull().sum()/ titanic_df.shape[0] * 100).drop(columns=[target_col]).sort_values(ascending=False)
# cabin has 77% nulls and embarked has 0.2%
```

### dtypes - column data types


Which data types would be appropriate for Linear Regression?


```python
# display data types
titanic_df.dtypes.sort_values(ascending=False)
```

### Unique elements



If the total number of unique items in a field equals the number of rows, that suggests those fields are unique IDs ( e.g. names, IDs ) and should be dropped.


```python
# number of unique elements in a column
(titanic_df.nunique() / titanic_df.shape[0] * 100).sort_values(ascending=False)
# PassengerId and Name are 100% unique values so those are indeed identifier columns. Ticket is 76% unique values so probably not a useful feature.
# Age and Fare have low unique values so they are OK to work with.
```




## Data Cleaning



### Backup #1


```python
# make backup
titanic_df_bak1 = titanic_df.copy()
```

```python
# restore from backup
titanic_df = titanic_df_bak1.copy()
```


### Keep only the target and two features


Select `Age` for the target and `Fare` for the feature

```python
# select the target and one feature
feature_col = 'Fare'
```

```python
# Verify the target column
titanic_df[target_col]
```

```python
# Verify the feature column
titanic_df[feature_col]
```

### Backup #2


```python
# make backup
titanic_df_bak2 = titanic_df.copy()
```

```python
# restore from backup
titanic_df = titanic_df_bak2.copy()
```

### Address any rows with nulls


- If the number of nulls in a column is low, it would be easiest to drop the row.
- If the number of nulls in a column is high, it would be easiest to drop the column.
- If the number of nulls in a column is between the low and high, one should impute values for the nulls.


```python
# Create a variable for the dataframe so it now only contains the Age (target) and Fare (feature) columns, but keeps the original titanic_df dataframe
titanic_df_working = titanic_df[[target_col, feature_col]]
titanic_df_working # Verify
```

```python
# Modify the dataframe to drop all rows across our dataframe that contain nulls when looking into our target column.
titanic_df_working = titanic_df_working.dropna(subset=target_col)
```

```python
# Verify the shape
titanic_df_working.shape
# Shape now shows 714 rows down from 891 rows
```

#### Impute

**No inputation needed. No nulls in our feature 'Fare' and we dropped all the null columns referencing our target 'Age' column.**


Impute with the median.


```python
# Calculate the median

```

```python
# Assign median to nulls

```

```python
# Verify there are no nulls

```

```python
# Verify median is about the same as before

```

## Exploratory Data Analysis



Linear Regression has two requirements:
1. the features are correlated with the target
1. the features are not correlated with each other


### correlated with the target


```python
# Calculate a correlation matrix for the target
corr_mat = titanic_df_working.corr()
corr_mat
```

```python
# Display the correlation matrix
sns.heatmap(
    corr_mat,
    square=True,
    annot=False,
    fmt='d',
    cbar=True,
    cmap='coolwarm',
    vmin=-1,vmax=1
)

plt.title('Correlation of Target and Feature');
```

Ranking doesn't make sense for a single feature.

```python
# Create a data series with features ranked by correlation

```





### independence of features: correlation

```python
# Calculate a correlation matrix for the features

```

```python
# Display the correlation matrix

```

```python
# Create a data series with features ranked by correlation

```

## Processing


### Prepare data frame

```python
# Split data frame into target and features
clean_target_col = 'Age'
clean_feature_col = 'Fare'
```

```python
# Verify features
titanic_df_working[clean_feature_col]
```

```python
# Verify target
titanic_df_working[clean_target_col]
```


### Multiple Cross Validation ( CV )


Create a CV loop with 100 iterations.  Capture performance metric in an array names `results`.
Use accuracy for the performance metric.

**GNB model doesn't make sense for this dataset because we have a correlation between our target and feature. Running a linear regression model for the multi CV and using MSE as our performance metric.


```python
# Multiple CV loop
n = 100
results = np.zeros(n)

for i in range(n):
    X_train, X_test, y_train, y_test = train_test_split(
        titanic_df_working[[clean_feature_col]],
        titanic_df_working[[clean_target_col]],
        test_size=0.25,
        random_state=None
    )

    # Create an empty GNB model
    modelLR = LinearRegression()

    # Train ( fit ) the model
    modelLR.fit( (X_train), y_train)

    # Make predicitons using the model
    pred_results = modelLR.predict(X_test)

    # calculate the accuracy
    results[i] = metrics.r2_score(y_test, pred_results)

results
```

#### Performance


We'll use RMSE as our performance metric.

```python
# Calculate mean of results
results.mean()
```

```python
# Display a histogram of the results
sns.histplot(results, kde=True, bins=20)
plt.title('Distribution of R² Scores Across 100 Iterations')
plt.xlabel('R² Score')
plt.ylabel('Count')
;
```


## Communication of Results


### Insights

```python

```

1. Mean
    - Mean of -0.0041 shows the model has virtually no predictive power
    - performes slightly worse than the average than simply predicting the mean of the target values

1. Most Results Close to Zero
    - Most of the \(R^2\) scores appear to fall between approximately -0.015 and 0.015.
    - The highest concentration is around 0 to 0.01.
    - This suggests that the model's performance was generally close to an \(R^2\) score of zero.

1. Some results are negative
    - Long tail to the left. Those negative \(R^2\) scores mean on those test splits the model performed worse than a baseline of the mean of actual target values.

1. Distribution skewed to the left
    - Negative skew (left skew) shows more scores cluster near zero.
      - model's performance is weak
      - little evidence of a strong prediction
    - Longer tail extends towards negative scores
      - smaller number of splits performed noticeable worse than otheres


## Conclusion:
- LR model looks to have weak predictive performance
- model's results vary depending on the train/test split


### Future work:
- The historgram alone doesn't tell us everything.
- For future, calculate the standard deviation, min, and max of the 100 scores. Why?
  - quanitfy typical performance
  - see how much the performance varies

```python
print("Mean R²:", np.mean(results))
print("Standard deviation:", np.std(results))
print("Minimum R²:", np.min(results))
print("Maximum R²:", np.max(results))
```

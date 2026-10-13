# Naive Bayes Classification to Predict Titanic Survival


**Deep Dive Coding Data Science Bootcamp Sample Project**

---
Robert Citek - October 5th, 2026



## Problem Definition

State the business problem. Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.  Mention which model will be used, if known.


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
from sklearn.naive_bayes import GaussianNB
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
titanic_df
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
titanic_df.describe()
```


### target value_counts()



The target should be the first field that you look at.  Its data type will tell you what kind of machine learning problem this is ( classification or regression. ) Any nulls should be culled.  You can check for any skew in the data set.

```python
# value_counts on target ; include nulls
titanic_df['Survived'].value_counts(dropna=False)
```


```python
# normalized value_counts on target ; include nulls
titanic_df['Survived'].value_counts(dropna=False, normalize=True) * 100
```

### Nulls



```python
# look for nulls
titanic_df.isnull().sum().sort_values(ascending=False)
```


```python
# look for nulls, displaying relative frequency
null_sum = titanic_df.isnull().sum()
null_rel_freq = null_sum / titanic_df.shape[0] * 100
null_rel_freq.sort_values(ascending=False)
```

### dtypes - column data types


Which data types would be appropriate for Gaussian Naive Bayes?


```python
# display data types
titanic_df.dtypes
```

### Unique elements



If the total number of unique items in a field equals the number of rows, that suggests those fields are unique IDs ( e.g. names, IDs ) and should be dropped.


```python
# number of unique elements in a column
titanic_df.nunique().sort_values(ascending=False)
```




## Data Cleaning



### Backup #1


```python
# make backup
titanic_df_bak = titanic_df.copy()
```

```python
# restore from backup
titanic_df = titanic_df_bak
```


### Keep only the target and two features


Select `Survived` for the target and `Age` and `Fare` for the features

```python
# select the target and one feature
titanic_df = titanic_df[['Survived','Age','Fare']]
titanic_df
```

### Backup #2


```python
# make backup
titanic_df_bak2 = titanic_df.copy()
```

```python
# restore from backup
titanic_df = titanic_df_bak2
```

### Address any rows with nulls


- If the number of nulls in a column is low, it would be easiest to drop the row.
- If the number of nulls in a column is high, it would be easiest to drop the column.
- If the number of nulls in a column is between the low and high, one should impute values for the nulls.



#### Impute


Impute with the median.


```python
# Calculate the median
titan_median = titanic_df['Age'].median()
titan_median
```

```python
# Assign median to nulls
titanic_df.fillna({'Age' : titan_median}, inplace=True)
titanic_df
```

```python
# Verify there are no nulls
titanic_df.isnull().sum().sort_values()
```

```python
# Verify median is about the same as before
titan_median
```

## Exploratory Data Analysis



GNB has two requirements:
1. the features are normally distributed
1. the features are not correlated with each other

### normal distribution of features: histogram


```python
# Display a histogram
titanic_df.hist() ;
```


### independence of features: correlation

```python
# Calculate a correlation matrix
iso_features = titanic_df[['Age', 'Fare']]

corr_mtrx = iso_features.corr()
```

```python
# Display the correlation matrix
corr_mtrx
```


```python
sns.heatmap(
    corr_mtrx,
    square=True,
    annot=True,
    fmt='2f',
    cbar=True,
)

plt.xlabel('Features')
plt.xlabel('Features')
```

## Processing


### Balance the data set


```python
# Display value counts of target
titanic_df['Survived'].value_counts()
```

```python
# Place minority class in one data frame
minority_class = 1
minority_df = titanic_df[titanic_df['Survived'] == minority_class]
minority_df
```

```python
# Create downsample of majority class in another data frame
# Put majority class into its own dataframe
majority_class = 0
majority_df = titanic_df[titanic_df['Survived'] == majority_class]

majority_downsampled = majority_df.sample(
    n=len(minority_df),
    random_state=42
)

print("Minority class:")
print(minority_df['Survived'].value_counts())

print("Downsampled majority class:")
print(majority_downsampled['Survived'].value_counts())
```

```python
# Combine data frames
balanced_df = pd.concat([minority_df, majority_downsampled])

balanced_df = balanced_df.reset_index(drop=True)
```

```python
# Verify target counts
balanced_df['Survived'].value_counts()
```


```python
# Split data frame into target and features
target_col = 'Survived'
target = titanic_df[ target_col]
features = titanic_df.drop(columns=[target_col])
```

```python
# Verify features
target
```

```python
# Verify target
features
```


### Multiple Cross Validation ( CV )


Create a CV loop with 100 iterations.  Capture performance metric in an array names `results`.
Use accuracy for the performance metric.


```python
# Multiple CV loop
n = 100
results = np.zeros(n)

for i in range(n):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=None
    )

    # Create an empty GNB model
    modelGNB = GaussianNB()

    # Train ( fit ) the model
    modelGNB.fit( (X_train), y_train)

    # Make predicitons using the model
    pred_results = modelGNB.predict(X_test)

    # calculate the accuracy
    results[i] = metrics.accuracy_score(y_test, pred_results)

results
```

#### Performance


We'll meassure accuracy, which is the proportions of the number correctly predicted, which is mathematically the same as one minus the proportion incorrectly predicted.

```python
# Calculate mean of results
results.mean()
```

```python
# Display a confusion matrix
mtrx = metrics.confusion_matrix(y_test, pred_results)

sns.heatmap(mtrx.T,
    square = True,
    annot  = True,
    fmt    = '2f',
    cbar   = False,
)
plt.xlabel('Actual')
plt.ylabel('Predicted')
```

```python
# Display a histogram of the results
pd.Series(results).hist()
```


## Communication of Results


Conclusion:


Future work:




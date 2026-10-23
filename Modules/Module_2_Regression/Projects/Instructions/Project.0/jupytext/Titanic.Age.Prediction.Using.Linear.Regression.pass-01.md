# Naive Bayes Classification to Predict Titanic Survival


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
titanic_df = pd.read_csv( url , usecols=['Age','Fare'])
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
titanic_df.describe().transpose()
```


### target value_counts()



The target should be the first field that you look at.  Its data type will tell you what kind of machine learning problem this is ( classification or regression. ) Any nulls should be culled.  You can check for any skew in the data set.

```python
# look for nulls in the target
target_col = titanic_df['Age']
target_col.value_counts(dropna=False)
```


```python
# Drop rows that contain nulls in the target
titanic_df = titanic_df.dropna(axis=0, subset=titanic_df.columns[0])
```

```python
# Verify that it removed the columns
titanic_df.shape
```

### Nulls



```python
# look for nulls
titanic_df['Fare'].isnull().sum()
# No nulls in the feature column
```


```python
# look for nulls, displaying relative frequency
titanic_df.isnull().mean().sort_values(ascending=False)
```

### dtypes - column data types


Which data types would be appropriate for Linear Regression?


```python
# display data types
titanic_df.dtypes
```

### Unique elements



If the total number of unique items in a field equals the number of rows, that suggests those fields are unique IDs ( e.g. names, IDs ) and should be dropped.


```python
# number of unique elements in a column
titanic_df.nunique().sort_values(ascending=False) / titanic_df.size * 100
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
target = titanic_df[['Age']]
feature = titanic_df[['Fare']]
```

```python
# Verify the target
target
```

```python
 # Verify the feature
 feature
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



#### Impute


No imputation needed. Already looked at the target and culled nulls for the working dataframe.




```python
# Calculate the median

```

```python
# Assign median to nulls


```

```python
# Verify there are no nulls
titanic_df.isnull().sum()
```

```python
# Verify median is about the same as before

```

## Exploratory Data Analysis



Linear Regression has two requirements:
1. the features are correlated with the target
1. the features are not correlated with each other


### ===START HERE=== correlated with the target


```python
# Calculate a correlation matrix for the target
corr_mat = titanic_df.corr()
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

plt.title('Correlation Between Target and Feature');
```

```python
# Create a data series with features ranked by correlation
```
### independence of features: correlation

```python
# Calculate a correlation matrix for the features
corr_mat_feature = feature.corr()
```

```python
# Display the correlation matrix
corr_mat_feature
```

```python
# Create a data series with features ranked by correlation

```

## Processing



## Residual analysis



## Communication of Results


Conclusion:


Future work:




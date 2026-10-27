<a href="https://colab.research.google.com/github/efitznm/Data-Science-Portfolio-C22/blob/main/Modules/Module_2_Regression/Projects/Instructions/Project.2-Housing/Housing-Pass-04.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


# Project 2 - Pass 04: Housing Price Prediction



The project should be done using Regression prediction techniques ( with feature scaling and regularization ) as covered in class.
The goal is to minimize the RMS ***percentage*** error ( root mean squared percentage error - RMSPE ) on your prediction of the house sales price.




Any of the available features can be used in the regression, but a minimum set of variables that do the job should be the ultimate goal.




Be sure to go through the whole data science process and document as such in your Jupyter notebook.




This project will have less direct "To Do" guidance and the progression through the data science process will be more individualized this time around.  We will talk about all the issues during class so you're not going to be out on a ledge with this one, I just want to move you towards performing a data science project on your own eventually.


There are two parts to this data set, both of which are on AWS:

- [Housing Data Dictionary]( https://ddc-datascience.s3.amazonaws.com/Projects/Project.2-Housing/Housing%20-%20Data%20Documentation.pdf )
- [Data Set]( https://ddc-datascience.s3.amazonaws.com/Projects/Project.2-Housing/Data/Housing.Data.csv)


```python
! curl -s -I https://ddc-datascience.s3.amazonaws.com/Projects/Project.2-Housing/Data/Housing.Data.csv
```

```python
! curl -s https://ddc-datascience.s3.amazonaws.com/Projects/Project.2-Housing/Data/Housing.Data.csv | head -2
```
## Problem Definition


### Business Problem

Target is *Sales Price* and feature is *Lot Frontage*

Help investors evaluate whether lot frontage might be a useful factor when assessing a property's potential market value.


### Data Science Problem

Supervised regression


## Data Collection/Sources


### Load Modules

```python
# Import the libraries and modules
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

### Load Data

```python
# Define the data's url as 'url'
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Project.2-Housing/Data/Housing.Data.csv'    # insert url here
url
```

```python
# Define the data frame as 'df'
df = pd.read_csv(url)
```

## Data Cleaning


### Overview

```python
# Display the head - 81 columns
df.head()
```

```python
# Display the tail - target is the last row
df.tail()
```

```python
# See the shape - 2637 rows, 81 columns
df.shape
```

```python
# See how many total entries
df.size
```

### Summary Statistics

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

```python
# Summary meta-data - our target 'SalePrice' is int64, our feature 'Lot Frontage' is float64
df.info()
```

```python
# See the summary statistics
df.describe().transpose()
```

### Target Nulls



The target should be the first field that you look at.  Its data type will tell you what kind of machine learning problem this is ( classification or regression. ) Any nulls should be culled.  You can check for any skew in the data set.

```python
# Define the name of the target column 'SalePrice' to work with it independently
target_col = 'SalePrice'
df[target_col]
```

```python
# Look for nulls in the target 'SalePrice' (dropping rows if 20% nulls or under) - no nulls in the target
df[target_col].isnull().sum() / df[target_col].shape[0] * 100

```


### Feature Nulls

```python
# Define the name of the feature column 'Lot Frontage' to work with it independently
feature_col = 'Lot Frontage'
df[feature_col]
```

```python
# Look for nulls in the feature 'Lot Frontage' (dropping rows if 20% nulls or under) - 17% are nulls
df[feature_col].isnull().sum() / df[feature_col].shape[0] * 100
```

### Identifier Columns

```python
# See proportion of unique values in the target column
(df[target_col].nunique() / df[target_col].shape[0] * 100) #.sort_values(ascending=False)
```

```python
# See proportion of unique values in the feature column
(df[feature_col].nunique() / df[feature_col].shape[0] * 100)
```

```python
# See for all data - PID is an identifier column
(df.nunique() / df.shape[0] * 100).sort_values(ascending=False)
```

```python
# See the data types of our target and feature - continuous are good for linear regression
df[[target_col, feature_col]].dtypes
```

### Backup #1

```python
# Make a backup
df_bak1 = df.copy()
```

```python
# Restore from backup
df = df_bak1.copy()
```

### Cull the Nulls

Feature column containing 17% nulls, so drop those rows in our dataset.

```python
# Drop null rows referencing the feature column
df = df.dropna(subset=feature_col)
```

```python
# Verify rows dropped
df.shape
```

### Backup #2

```python
# Make a backup
df_bak2 = df.copy()
```

```python
# Restore from backup
df = df_bak2.copy()
```

### Working Data Frame

```python
# Define our target and feature columns as a data frame to work with 'iso_df'
iso_df = df[[target_col, feature_col]].reset_index(drop=True)
```

```python
# Verify the working data frame
iso_df
```

## EDA (Exploratory Data Analysis)


### Linear Regression Check

Linear Regression has two requirements:
1. the features are correlated with the target
1. the features are not correlated with each other


#### Correlation with Target

```python
# Calculate a correlation matrix for the target
corr_mat = iso_df.corr()
corr_mat
```

##### Correlation matrix

```python
# Display the correlation matrix - check if feature correlated with the target
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

Feature is highly correclated with the target.


##### Correlation Ranking
Ranking doesn't make sense for a single feature.

```python
# Create a data series with features ranked by correlation

```

### Independence of Features
Doesn't make sense for a single feature.

```python
# Calculate a correlation matrix for the features

```

```python
# Display the correlation matrix

```

```python
# Create a data series with features ranked by correlation

```

## Data Processing


Prepare Data Frame (Target and Features)

```python
# Define out the target and feature columns from our cleaned working data frame
iso_df_target = 'SalePrice'
iso_df_feature = 'Lot Frontage'
```

#### Multi CV

```python
# Multiple CV loop
n = 100
results = np.zeros(n)

for i in range(n):
    X_train, X_test, y_train, y_test = train_test_split(
        iso_df[[iso_df_feature]],
        iso_df[[iso_df_target]],
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


## Data Visualization

```python

```

## Communication of Results

```python
print("Mean R²:", np.mean(results))
print("Standard deviation:", np.std(results))
print("Minimum R²:", np.min(results))
print("Maximum R²:", np.max(results))
```

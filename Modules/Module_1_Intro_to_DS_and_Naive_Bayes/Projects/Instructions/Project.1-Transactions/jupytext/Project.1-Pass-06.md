<a href="https://colab.research.google.com/github/efitznm/Data-Science-Portfolio-C22/blob/main/Modules/Module_1_Intro_to_DS_and_Naive_Bayes/Projects/Instructions/Project.1-Transactions/Project.1-Pass-06.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


# Project.1_Pass-06, Transaction Data





## Problem Definition



State the business problem. Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.


### Business Problem
We want to predict whether or not a future customer will make a transaction based on their transaction data. The goal might be to increase customer transactions by sending targeted ads/deals to customers who are not likely to make a transaction on a given day.


### Data Science Problem
Supervised binary classification problem.


## Data Collection/Sources



### Load Pandas, Numpy, and Matplotlib modules

```python
# Load Pandas, Numpy, and Matplotlib modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets, metrics, model_selection
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
```

### Load data Train.csv from AWS S3[link text](https://)

```python
# Load data Train.csv from AWS S3
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Project.1-Transactions/Data/Transaction.train.csv'
url

df = pd.read_csv( url )

df
```

```python
# Look at the first few rows
df.head()
```

```python
# Look at the last few rows
df.tail()
```

```python
# Shape of the dataframe
df.shape
```

```python
# Size of the dataframe
df.size
```

### Summary Meta-Data

```python
# Display meta-data
df.info()
```

### Summary Statistics

```python
# Display summary statistics
df.describe()
```

### NOTE: the ‘target’ column indicates a successful transaction (‘1’) or a no-transaction (‘0’). Verify these are the only values in that column, and that there are no nulls.


### Target Value Counts

Binary numerical classification and no nulls.

```python
# Value_counts on target ; include nulls
df['target'].value_counts(dropna=False)
```

### Normalize The Data

```python
# Normalized value_counts on target ; include nulls
df['target'].value_counts(dropna=False, normalize=True)
```

### Nulls
No imputation because no nulls.

```python
# Look for nulls
df.isnull().sum().sort_values(ascending=False)
```

Column Data Types

```python
# Display data types
df.dtypes
```

```python
# Number of unique elements in a column
df.nunique().sort_values(ascending=False)
```

## Data Cleaning



### Backup #1

```python
# Make backup
df_bak = df.copy()
```

```python
# Restore from backup
df = df_bak.copy()
```

### Keep Only the Target and Features

```python
# Select the target and features
df = df[['target'] + [f'var_{i}' for i in range(50)]]
df
```

### Backup #2

```python
# Make 2nd backup
df_bak_2 = df.copy()
```

```python
# Restore from 2nd backup
df = df_bak_2.copy()
```

## Exploratory Data Analysis



### Normal Distribution of Features: Histogram & Box Plot

```python
# Display a histogram
df.hist() ;
```

```python
# Display a box and whisker plot
df.plot(kind='box') ;
```

### Correlation: Independence of Features

```python
# Calculate a correlation matrix
iso_features = df[['target'] + [f'var_{i}' for i in range(50)]]

corr_mat = iso_features.corr()

corr_mat
```

```python
# Heatmap of the correlation matrix
sns.heatmap(
    corr_mat,
    square=True,
    annot=True,
    fmt='2f',
    cbar=True,
)

plt.xlabel('Features')
plt.xlabel('Features')
```

### Create Two Dataframes (Copies, Not Slices)
1. Successful Transactions
1. Unsuccessful Transactions

```python
# Create a dataframe of successful transactions
successful_df = df[df['target'] == 1].copy()

# Create a dataframe of unsuccessful transactions
unsuccessful_df = df[df['target'] == 0].copy()
```

## Processing




### Single Cross Validation ( CV )


#### Two data sets with target and features

```python
# Split data frame into target and features
target_col = 'target'
target = df[ target_col]
features = df.drop(columns=[target_col])
```

```python
# Display the target
target
```

```python
# Display the features
features
```

#### Define an empty Gaussian Naïve Bayes model using Sklearn

```python
# Create an empty GNB model
modelGNB = GaussianNB()
```

#### Divide the two data sets you created in step 10 into training and testing subsets, i.e. train-test split.

```python
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
     features,
     target,
     test_size=0.25
)
```

```python
# Verify shape of each data set from Train Test Split
for i in [ X_train, X_test, y_train, y_test ]:
  print(i.shape)
  print(type(i))
```

#### Train ( fit ) the model using the training subset of the dataset.

```python
# Train ( fit ) the model
modelGNB.fit( (X_train), y_train)
```

#### Test the model ( predict ) using the testing subset of the dataset.

```python
# Make predicitons using the model
pred_results = modelGNB.predict(X_test)
pred_results * 1000
```

#### Calculate and report the accuracy.

```python
# Display accuracy
acc = metrics.accuracy_score( y_test,pred_results )
acc * 100
```

#### Present a confusion matrix and heatmap

```python
# Display a confusion matrix and heatmap
mat =  metrics.confusion_matrix(
    y_test,
    pred_results
)

sns.heatmap( mat.T,
    square=True,
    annot=True,
    fmt = "2f",
    cbar = False,
)
plt.xlabel('Actual')
plt.ylabel('Predicted');
```

#### Present the results of a Classification Report (sklearn.metrics.classification_report).

```python
# Display a classification report
print(metrics.classification_report(y_test, pred_results))
```

### Multiple Cross-Validation (CV)

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

### Accuracy Comparison
Accuracies are both 91% for single and multi CVs

```python
# Compare the accuracies by creating variables for each
acc_single_cv = acc
acc_multi_cv = results[i]

display('Single CV:', acc_single_cv)
display('Multi CV:', acc_multi_cv)
```

### Balance Dataset

The training data is very skewed towards non-successful transactions (about 90% of the training data has ‘target’==0).


#### Remove enough non-successful transaction rows so that your remaining training data is 50%/50% split between successful and non-successful transactions.

```python
# Display value counts of the target
df['target'].value_counts()
```

```python
# Place minority class in one dataframe
successful_df
```

```python
# Put majority class into its own dataframe
unsuccessful_df
```

```python
# Create downsample of majority class in another data frame
majority_downsampled = unsuccessful_df.sample(
    n=len(successful_df),
    random_state=42
)

# Display the downsample to verify
print("Minority class:")
print(successful_df['target'].value_counts())

print("Downsampled majority class:")
print(majority_downsampled['target'].value_counts())
```

```python
# Combine the dataframes
balanced_df = pd.concat([successful_df, majority_downsampled])

# Reset the index
balanced_df = balanced_df.reset_index(drop=True)
```

```python
# Verify target counts
balanced_df['target'].value_counts()
```

#### Backup #3

```python
# Create 3rd backup
df_bak_3 = df.copy()
```

```python
# Restore from backup
df = df_bak_3.copy()
```

#### Repeat Multi Cross-Validation

```python
# Split data frame into target and features
target_col = 'target'
target = balanced_df[ target_col]
features = balanced_df.drop(columns=[target_col])
```

```python
# Display the target
target
```

```python
# Display the features
features
```

```python
# Multiple CV loop
n = 100
balanced_df_results = np.zeros(n)

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
    balanced_df_results[i] = metrics.accuracy_score(y_test, pred_results)

balanced_df_results
```

#### Mean accuracy for balanced dataset

```python
# Report what your mean accuracy for this balanced data set
display(balanced_df_results[i])
```

#### Histogram of the accuracy scores

```python
pd.Series(balanced_df_results).hist() ;
```

What do you notice about the distribution of accuracy scores?

There is a negative skew.


#### Confusion Matrix and Heatmap

```python
mat =  metrics.confusion_matrix(
    y_test,
    pred_results
)

sns.heatmap( mat.T,
    square=True,
    annot=True,
    fmt = "2f",
    cbar = False,
)
plt.xlabel('Actual')
plt.ylabel('Predicted');
```

#### Classification Report

```python
print(metrics.classification_report(y_test, pred_results))
```

What do you notice? That is, how do the different metrics compare to each other?


## Data Visualization/Communication of Results



### Data Visualization

```python
# Display a histagram of the results
pd.Series(balanced_df_results).hist() ;
```

### Conclusion


- Using all var columns and balancing the dataset gave us 91% accuracy, same as with previous Pass-05
- Bad at predicting no-transactions
- OK at predicting a successful transaction
- Low false positive (we predict customer will make a transaction but customer does not)
- High false negative (we predict customer will not make a transaction but customer does)


### Future work



- For Pass-07,
  - Use big dataset
  - Use only the first 15 columns
  - Use ony the first 10,000 rows

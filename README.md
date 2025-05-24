# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
### Developed by : Vineela Shaik
### Reg No : 212223040243
```
# FEATURE SCALING
import pandas as pd
from scipy import stats
import numpy as np
```


```
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/c6e3ed69-f2db-41cc-a492-c08ea52a5bcf)

```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/f0df2849-f955-4f8f-9b19-d687f256cfb5)

```
df.dropna()
```
![image](https://github.com/user-attachments/assets/778658cd-8745-457f-9992-337a009e6734)

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
# This is typically used in feature scaling,
#particularly max-abs scaling, which is useful
#when you want to scale data to the range [-1, 1]
#while maintaining sparsity (often used with sparse data).
```
![image](https://github.com/user-attachments/assets/ad078354-a7ec-4824-b193-abab49272d93)

```
# Standard Scaling
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
![image](https://github.com/user-attachments/assets/c3997748-b604-4f9b-b61f-6d7a180dabab)

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/28fd338e-68a1-4892-80f8-e9f457fc1024)

```
#MIN-MAX SCALING:
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/1cb180c2-cb10-42ad-9daa-392b59cb8a5b)

```
#MAXIMUM ABSOLUTE SCALING:

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/7f5672be-6108-45a7-95ec-b7bfb35ef661)

```
#ROBUST SCALING

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
![image](https://github.com/user-attachments/assets/b97e589b-33bf-4dab-80e1-3c000ce648ed)

```
#FEATURE SELECTION:

df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![image](https://github.com/user-attachments/assets/4279091f-4369-4d81-a1fd-f3371fc3fcdc)

```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/99195e69-2e38-46c3-a32c-050f6fdd408d)

```
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
#In feature selection, converting columns to categorical helps certain algorithms
# (like decision trees or chi-square tests) correctly understand and
 # process non-numeric features. It ensures the model treats these columns as categories,
  # not as continuous numerical values.
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/1d878f9e-a8d1-4bb1-a6da-e0fde5ab799b)

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/fa7c4e39-7ab5-4b1f-a0db-1b8a26a7ed71)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
#X contains all columns except 'SalStat' — these are the input features used to predict something.
#y contains only the 'SalStat' column — this is the target variable you want to predict.
```

```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/61cd0112-10dd-4d15-a08b-547b27d8cfc8)

```
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()

```
![image](https://github.com/user-attachments/assets/6627af6e-b26d-4cf8-a912-2886c50f95b2)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/bea4a1e0-e442-47ea-be3a-eebf0b090434)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/4060b42b-aee8-4257-b407-74810343cf1e)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/a07e6b66-029c-41cd-a8d1-430bb40bf77c)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/992d24b7-91ee-4dde-8983-7bc9bf28c112)

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/c0783a7e-bf1b-4d8a-9bf3-b70a91d4503e)

```
!pip install skfeature-chappers
```
![image](https://github.com/user-attachments/assets/b6371826-058d-4285-acdb-5ff9769562a0)

```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

```
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
```

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/09b3c379-fee0-4cf2-850b-c3e4e59df005)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```

```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
```

```
selected_features_anova = X.columns[selector_anova.get_support()]
```

```
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![image](https://github.com/user-attachments/assets/f1facfde-5bf9-4645-a188-23c9e477c141)

```
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')

```

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/4b2fce23-14fc-449a-9fc6-4f229d6d34d0)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```

```
logreg = LogisticRegression()
```

```
n_features_to_select =6
```

```
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![image](https://github.com/user-attachments/assets/cb365aa6-c949-4af8-9e77-6f937d679e69)



# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset

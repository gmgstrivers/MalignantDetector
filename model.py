# Breast Cancer Prediction

# #Breast cancer is the most common type of cancer in women. When cancers are
# found early, they can often be cured. There are some devices that detect the
# breast cancer but many times they lead to false positives, which results is
# patients undergoing painful, expensive surgeries that were not even necessary.
# These type of cancers are called benign which do not require surgeries and we
# can reduce these unnecessary surgeries by using Machine Learning. We take a
# dataset of the previous breast cancer patients and train the model to predict
# whether the cancer is benign or malignant. These predictions will help doctors
# to do surgeries only when the cancer is malignant, thus reducing the
# unnecessary surgeries for woman.

# For building the project we have used Wisconsin Breast cancer data which has
# 569 rows of which 357 are benign and 212 are malignant.

# load the dataset
import pandas as pd

df = pd.read_csv("./breast_cancer.csv")

# display first 5 rows
df

# display last 5 rows
df.tail()

# dipslay 5 smaples randomly
df.sample(5)

# Display No of observation and columns
df.shape

# display the columns
df.columns

# display No of people  benign or malignant
df['diagnosis'].value_counts()

# Display in a barchart
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x=df['diagnosis'])
plt.show()

# Display the iabove information in percentage
import matplotlib.pyplot as plt

z = plt.pie(df['diagnosis'].value_counts(), \
            labels=['Benign', 'Malignant'], autopct='%0.2f%%', shadow=True, textprops={'fontsize': 20})

# REMOVE  'id'  and 'Unnamed: 32'  column

df1 = df.drop(columns=['Unnamed: 32', 'id'])  # Dropping irrelevant columns

df1.head(2)

df1.shape

#  Find Missing values in each column

df1.isnull().sum()

# Dropping Rows having 0 in Dataset

import numpy as np

df1 = df1.replace(0, np.nan)
df2 = df1.dropna()
df2.shape

# Droppping Duplicate Values
df3 = df2.drop_duplicates()
df3.shape

## Now  display No of people  benign or malignant
df3['diagnosis'].value_counts()

# Now convert the diagnosis column to numeric
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df3['diagnosis'] = le.fit_transform(df3['diagnosis'])

df3.sample(5)  # B---0    M---1

# Find Corerelation between features

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20, 20))
sns.heatmap(df3.corr(), annot=True)
plt.show()

# Dropping columns having co-relation of 88 and higher


import numpy as np

corr_matrix = df3.corr().abs()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df3 = corr_matrix.mask(mask)

to_drop = [x for x in tri_df3.columns if any(tri_df3[x] > 0.88)]

df3 = df3.drop(to_drop, axis=1)

print(f"The reduced dataframe has {df3.shape[1]} columns.")

df3.shape

plt.figure(figsize=(20, 20))
sns.heatmap(df3.corr(), annot=True)
plt.show()

# Separate input and output


X = df3.drop(columns=['diagnosis'])
Y = df3['diagnosis']

Y

# Splitting for train and test
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

X_train.shape

X_test.shape

# Scale down the features  by using Normalization


# Scaling
from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)



import pickle
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=40)
# Train the model
rf.fit(X_train, Y_train)

pickle.dump(rf, open("model.pkl", "wb"))




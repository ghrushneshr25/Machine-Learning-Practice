from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # Encoding Dep Variable
from sklearn.preprocessing import OneHotEncoder  # Encoding independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # HANDLING MISSING DATA


"----------IMPORTING LIBRARIES-----------"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"----------------------------------------"

"----------IMPORTING DATASET-------------"
dataset = pd.read_csv("Data.csv")  # read dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
"print(X)"
"print(Y)"
"----------------------------------------"

"-----------HANDLING MISSING DATA--------"
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# ALL MISSING VALUES IN X ARE REPLACED BY  AVG VALUE OF THAT PARTICULAR COLUMN
"print(X) "
"----------------------------------------"

"-------ENCODING CATEGORICAL DATA--------"

"from sklearn.compose import ColumnTransformer"

# Encoding independent Variable
"from sklearn.preprocessing import OneHotEncoder "
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
"print(X)"

# Encoding Dependent Variable
"from sklearn.preprocessing import LabelEncoder"

le = LabelEncoder()
Y = le.fit_transform(Y)
"print(Y)"
"----------------------------------------"

"SPLITTING DATASET INTO TEST SET AND TRAINING SET"
"from sklearn.model_selection import train_test_split "
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1)
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)
"----------------------------------------"

"FEATURE SCALING"
"from sklearn.preprocessing import StandardScaler"

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
#print(X_train)
X_test[:, 3:] = sc.transform(X_test[:, 3:])
#print(X_test)

"----------------------------------------"

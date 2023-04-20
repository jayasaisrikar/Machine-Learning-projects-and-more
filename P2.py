#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

# load dataset
df = pd.read_csv("C:\\Users\\JAYA SAI SRIKAR\\OneDrive\\Desktop\\Projects\\P2\\P2\\nba2021.csv")

# drop irrelevant columns
cols_to_drop = ["Rk", "Player", "Age", "Tm"]
for col in cols_to_drop:
    if col in df.columns:
        df = df.drop(col, axis=1)

# map positions to integers
position_map = {"SG": 0, "PG": 1, "SF": 2, "PF": 3, "C": 4}
df["Pos"] = df["Pos"].map(position_map)

# split data into train and test sets
X = df.drop("Pos", axis=1)
y = df["Pos"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# fit decision tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# predict test set and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# calculate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5])
print(f"Confusion Matrix:\n{cm}")

# apply 10-fold stratified cross-validation
skf = StratifiedKFold(n_splits=10)
fold_accuracy = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracy.append(accuracy)
    print(f"Fold Accuracy: {accuracy}")
    
avg_accuracy = sum(fold_accuracy) / len(fold_accuracy)
print(f"Average Accuracy: {avg_accuracy}")


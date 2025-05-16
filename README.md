# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries like sklearn, pandas, and matplotlib.

2.Load the Iris dataset and convert it into a DataFrame.

3.Split the data into features (X) and target (Y).

4.Divide data into training and testing sets.

5.Train the model using SGDClassifier on the training data.

6.Predict and evaluate using accuracy score and confusion matrix.



## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: AMAN ALAM
RegisterNumber:  212224240011

Program to implement the prediction of iris species using SGD Classifier.
Developed by: Akash M
RegisterNumber:  212224230013
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
iris = load_iris()

# Create pandas dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())
# Split the data into features (X) and target (Y)
x = df.drop('target', axis=1)
y = df['target']
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Create SGD Classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

# Train the classifier on the training data
sgd_clf.fit(x_train, y_train)
# Make predictions on the testing data
y_pred = sgd_clf.predict(x_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
# Calculate the confusion matrix
cf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cf)

*/
```

## Output:
![image](https://github.com/user-attachments/assets/a00d9ccc-3f80-4c9a-8061-a22a3751ee43)

![image](https://github.com/user-attachments/assets/808d73e6-9548-4326-9394-b0b89bf5c1b4)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

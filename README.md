# 4-dimensional Star Classification by Wavelength Values
This Notebook is created to classify 4-dimensional vectors based on whether they belong to a star. The Notebook has a lot of graphs to visualize the problem and to find the best approach with the model & data. ```imblearn```'s ```Pipeline``` is used to train and fit the model to data to avoid data leakage and for easier hyperparameter tuning.

# Pipeline
```pipe = Pipeline([
    ('imputer', KNNImputer(n_neighbors=7, weights="distance")),   
    ('resample', SMOTE(random_state = 42)),
    ('scaler', MinMaxScaler()),
    ('dt', DecisionTreeClassifier())
    ])
```
## 1. Imputer
Impute NaN values. We will use KNNImputer, because there is a connection between vector points and their target values.


## 2. Resample
Our dataset is imbalanced. To classify minor classes correctly, we will resample our data using SMOTE.


## 3. Scaler
Our data will be scaled to result in a better classification. 


# Visualizing the Tree
![image](https://github.com/nazayis/star-classification/assets/69856039/9e56dfed-2cae-4ab3-a5ab-609593ea4845)

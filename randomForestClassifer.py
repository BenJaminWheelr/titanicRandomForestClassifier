import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('cleanedData.csv')

X = data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked', 'Wealth', 'Importance']]
Y = data['Survived']

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

paramGrid = {
    'n_estimators': [5, 10, 50, 100, 1000, 10000],
    'max_depth': [None, 5, 10, 15, 25, 30],
    'min_samples_split': [2, 5, 10]
}
gridSearch = GridSearchCV(estimator=rf, param_grid=paramGrid, cv=5, scoring='accuracy')

gridSearch.fit(xTrain, yTrain)

print(f"BEST HYPERPARAMETERS: {gridSearch.best_params_}")
print(f"BEST ACCURACY: {gridSearch.best_score_}")

# BEST HYPERPARAMETERS: {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 50}
# BEST ACCURACY: 0.8314192849404115
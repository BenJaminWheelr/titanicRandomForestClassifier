import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv('cleanedData.csv')
testData = pd.read_csv('cleanedDataTest.csv')


X = data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked', 'Wealth', 'Importance']]
Y = data['Survived']

xTestFinal = testData[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked', 'Wealth', 'Importance']]


totalAccuracy = 0
bestAccuracy = 0
bestModel = None
for i in range(100):
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=i)
    rf = RandomForestClassifier(n_estimators=50, random_state=i, max_depth=5, min_samples_split=5)

    rf.fit(xTrain, yTrain)

    yPredictions = rf.predict(xTest)
    score = accuracy_score(yTest, yPredictions)
    if score > bestAccuracy:
        bestModel = rf
        bestAccuracy = score
    totalAccuracy += score

avgAccuracy = totalAccuracy/100
print(avgAccuracy)
print(bestAccuracy)

yPredictionsTest = bestModel.predict(xTestFinal)

finalPredictions = pd.DataFrame({
    'PassengerId': testData['PassengerId'],
    'Survived': yPredictionsTest
})

finalPredictions.to_csv('submission.csv', index=False)
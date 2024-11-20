import pandas as pd

data = pd.read_csv('test.csv')

# print(data.describe())
# print(data['Age'].describe())
# print(data['Cabin'].describe())
def genderToBoolean(row):
    if row['Sex'] == 'male':
        return 1
    else:
        return 0


def replaceNanWithMedian(row):
    if pd.isna(row['Age']):
        return data['Age'].median()
    else:
        return row['Age']


def replaceEmbarkedWithId(row):
    embarkedLoc = row['Embarked']
    if embarkedLoc == 'C':
        return 0
    elif embarkedLoc == 'Q':
        return 1
    else:
        return 2


def setWealth(row):
    return row['Fare'] // row['Pclass']


def familyImportance(row):
    return row['Sex'] * (row['Parch'] + row['SibSp'])




data['Sex'] = data.apply(genderToBoolean, axis=1)
data['Age'] = data.apply(replaceNanWithMedian, axis=1)
data['Embarked'] = data.apply(replaceEmbarkedWithId, axis=1)
data['Wealth'] = data.apply(setWealth, axis=1)
data['Importance'] = data.apply(familyImportance, axis=1)

data.to_csv('cleanedDataTest.csv', index=False)
# Imports for proper work
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Cols for preprocessing our data with LabelEncoder()
cols = ['Sex', 'Embarked']
#LabelEncoder for removing strings from datasets
le = preprocessing.LabelEncoder()
#Creating datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


male_median = train_data['Age'].loc[(train_data.Sex == 'male')].median()
female_median = train_data['Age'].loc[(train_data.Sex == 'female')].median()


def clean(data):
    # Sum of every person's siblings and parents
    data['Relations'] = data['SibSp'] + data['Parch'] + 1
    # Removing useless columns from datasets
    data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'])
    # Filling Na/ Nan values
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    data['Embarked'] = data['Embarked'].fillna('S')

    # Convert strings to digits with LabelEncoder.fit_transform
    for col in cols:
        data[col] = le.fit_transform(data[col])

    data['Age'].loc[data.Sex == 0] = data['Age'].fillna(female_median)
    data['Age'].loc[data.Sex == 1] = data['Age'].fillna(male_median)

    # Making some new columns for better predictions
    data['Chance to die'] = data['Sex'] * data['Age']
    data['Chance to leave'] = data['Pclass'] + data['Sex']
    data['Relation survival'] = data['Embarked'] * data['Relations']
    data['Poverty'] = data['Pclass'] * data['Embarked']
    data['Women by relations'] = data['Relations'] * data['Sex']
    data['Women by chance'] = data['Chance to die'] * data['Age']
    data['Poverty by Embarked'] = data['Poverty'] * data['Embarked']

    return data


# Preprocessing our data
clean_train = clean(train_data)
clean_test = clean(test_data)

y = clean_train['Survived']
X = clean_train.drop(columns=['Survived'])

# Splitting data for test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

# Creating Logistic Regression model
clf = LogisticRegression(max_iter=2000).fit(X_train, y_train)
# Predictions on our train data
prediction = clf.predict(X_test)
print(f'Logreg accuracy: {accuracy_score(y_test, prediction)}')

# Predictions on our test data
k_predictions = clf.predict(clean_test)

# Creating pd.DataFrame to write it in csv file for submission
ids = test_data['PassengerId']
df = pd.DataFrame({'PassengerId': ids.values, 'Survived': k_predictions})

df.to_csv('submission.csv', index=False)

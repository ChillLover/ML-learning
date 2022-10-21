import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


le = preprocessing.LabelEncoder()
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_id = test_data['PassengerId']
cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
feature = ['sqrt']
crit = ['entropy']

params = {'max_depth': range(1, 100, 10),
          'max_features': feature,
          'criterion': crit,
          'n_estimators': [300],
          'min_samples_split': range(2, 150, 10),
          'min_samples_leaf': range(1, 150, 10)}


def cleaning(data):
    data = data.drop(columns=['PassengerId', 'Cabin', 'Name'])

    data['HomePlanet'] = le.fit_transform(data['HomePlanet'])
    data['CryoSleep'] = le.fit_transform(data['CryoSleep'])
    data['Destination'] = le.fit_transform(data['Destination'])
    data['VIP'] = le.fit_transform(data['VIP'])

    for col in cols:
        data[col] = data[col].fillna(data[col].median())

    data['Total fare'] = data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck']
    data['Chance'] = data['VIP'] * data['Age'] * 10

    return data


cleaned_train = cleaning(train_data)
cleaned_train['Transported'] = le.fit_transform(cleaned_train['Transported'])
cleaned_test = cleaning(test_data)

y = cleaned_train['Transported']
X = cleaned_train.drop(columns=['Transported'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier()
search = GridSearchCV(clf, params, n_jobs=-1)
search.fit(X_train, y_train)
best_tree = search.best_estimator_

k_predictions = best_tree.predict(cleaned_test)

df = pd.DataFrame({'PassengerId': test_id.values,
                   'Transported': k_predictions})

df['Transported'] = df['Transported'].astype(bool)

df.to_csv('k_data.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

pd.set_option('display.max_columns', None)


def will_you_survive(model, pclass, sex, age):
    user = np.array([pclass, sex, age]).reshape(1, 3)
    return model.predict(user)


def main():
    ############## processing ####################
    data = pd.read_excel('titanic.xls')
    data = data.drop(['name', 'sibsp', 'parch', 'ticket', 'fare',
                      'cabin', 'embarked', 'boat', 'body', 'home.dest'], axis=1)
    data = data.dropna()
    data['sex'] = data['sex'].replace('male', 1)
    data['sex'] = data['sex'].replace('female', 0)
    X = data.drop(['survived'], axis=1)
    y = data['survived']

############## training ####################
    model = KNeighborsClassifier()
    model.fit(X, y)
    print('precision: ', model.score(X, y))
    print(will_you_survive(model, 3, 1, 23))


if __name__ == "__main__":
    main()

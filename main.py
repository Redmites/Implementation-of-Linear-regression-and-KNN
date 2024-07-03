from functools import reduce

import pandas as pd
import scipy
import seaborn
import sklearn.metrics
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from math import sqrt


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distance(self, X1, X2):
        distance = scipy.spatial.distance.euclidean(X1, X2)
        return distance

    def predict(self, X_test):
        final_output = []
        for i in range(len(X_test)):
            d1 = []
            d2 = []
            sum = 0
            for j in range(len(self.X_train)):

                dist = self.distance(self.X_train.iloc[j] , X_test.iloc[i])
                d1.append(dist)
                d2.append(j)
            d1, d2 = (list(t) for t in zip(*sorted(zip(d1, d2))))
            i = 0
            while (i<self.k):
                if (self.y_train.iloc[d2[i]] == "muffin"):
                    sum = sum+1
                else:
                    sum = sum-1
                i = i+1

            if (sum > 0):
                final_output.append("muffin")
            else:
                final_output.append("cupcake")

        return final_output

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return (predictions == y_test).sum() / len(y_test)


class LinearRegressionGradientDescent:
    def __init__(self):
        self.coeff = None
        self.features = None
        self.target = None
        self.mse_history = None

    def set_coefficients(self, *args):
        self.coeff = np.array(args).reshape(-1, 1)

    def cost(self):
        predicted = self.features.dot(self.coeff)
        s = pow(predicted - self.target, 2).sum()
        return (0.5 / len(self.features)) * s

    def predict(self, features):
        features = features.copy(deep=True)
        features.insert(0, 'c0', np.ones((len(features), 1)))
        features = features.to_numpy()
        return features.dot(self.coeff).reshape(-1, 1).flatten()

    def gradient_descent_step(self, learning_rate):
        predicted = self.features.dot(self.coeff)
        s = self.features.T.dot(predicted - self.target)
        gradient = (1. / len(self.features)) * s
        self.coeff = self.coeff - learning_rate * gradient
        return self.coeff, self.cost()

    def perform_gradient_descent(self, learning_rate, num_iterations=100):
        # Istorija Mean-square error-a kroz iteracije gradijentnog spusta.
        self.mse_history = []
        for i in range(num_iterations):
            _, curr_cost = self.gradient_descent_step(learning_rate)
            self.mse_history.append(curr_cost)
        return self.coeff, self.mse_history

    def fit(self, features, target):
        self.features = features.copy(deep=True)
        # Pocetna vrednost za koeficijente je 0.
        # self.coeff - dimenzije ((n + 1) x 1)
        coeff_shape = len(features.columns) + 1
        self.coeff = np.zeros(shape=coeff_shape).reshape(-1, 1)
        # Unosi se kolona jedinica za koeficijent c0,
        # kao da je vrednost atributa uz c0 jednaka 1.
        self.features.insert(0, 'c0', np.ones((len(features), 1)))
        # self.features - dimenzije (m x (n + 1))
        self.features = self.features.to_numpy()
        # self.target - dimenzije (m x 1)
        self.target = target.to_numpy().reshape(-1, 1)



def co2Emissions():
    # 1. Учитавање скупа података и приказ првих пет редова у табели.

    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", None)
    data = pd.read_csv("fuel_consumption.csv")
    print (data.head())

    # 2. Приказ концизних информација о садржају табеле и статистичких информација о свим атрибутима скупа података.

    print(data.info())
    print(data.describe(include=[object]))

    # 3. Елиминисање примерака са недостајућим вредностима атрибута или попуњавање недостајућих вредности на основу вредности атрибута осталих примерака.

    data.TRANSMISSION = data.TRANSMISSION.fillna(data.TRANSMISSION.mode()[0])
    #data = data.where(data.ENGINESIZE.notnull())
    #data = data.where(data.FUELTYPE.notnull())

    #data = data.dropna(axis=0, subset=["ENGINESIZE"])
    #data = data.dropna(axis=0, subset=["FUELTYPE"])

    data = data[pd.notnull(data["ENGINESIZE"])]
    data = data[pd.notnull(data["FUELTYPE"])]

    data = data.reset_index()



    print(data.info())
    print(data.describe())

    # 4. Графички приказ зависности континуалних атрибута коришћењем корелационе матрице

    df = pd.DataFrame(data, columns=["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY","FUELCONSUMPTION_COMB","FUELCONSUMPTION_COMB_MPG","CO2EMISSIONS"])
    corr_matrix = df.corr()
    sn.heatmap(corr_matrix, annot=True, square=True, fmt='.2f')
    plt.show()

    # 5. Графички приказ зависности излазног атрибута од сваког улазног континуалног атрибута расејавајући тачке по Декартовом координатном систему

    df = pd.DataFrame(data)
    ax = df.plot.scatter(x="ENGINESIZE", y="CO2EMISSIONS", color="DarkBlue", label="ENGINESIZE")
    df.plot.scatter(x="CYLINDERS", y="CO2EMISSIONS", color="DarkGreen", label="CYLINDERS", ax=ax)
    df.plot.scatter(x="FUELCONSUMPTION_CITY", y="CO2EMISSIONS", color="Yellow", label="FUELCONSUMPTION_CITY", ax=ax)
    df.plot.scatter(x="FUELCONSUMPTION_HWY", y="CO2EMISSIONS", color="Red", label="FUELCONSUMPTION_HWY", ax=ax)
    df.plot.scatter(x="FUELCONSUMPTION_COMB", y="CO2EMISSIONS", color="Brown", label="FUELCONSUMPTION_COMB", ax=ax)
    df.plot.scatter(x="FUELCONSUMPTION_COMB_MPG", y="CO2EMISSIONS", color="Purple", label="FUELCONSUMPTION_COMB_MPG", ax=ax)
    plt.show()

    # 6. Графички приказ зависности излазног атрибута од сваког улазног категоричког атрибута користећи одговарајући тип графика.

    df = pd.DataFrame(data)

    seaborn.boxenplot(data=df, x="FUELTYPE", y="CO2EMISSIONS", color="Red")
    plt.show()
    seaborn.boxenplot(data=df, x="TRANSMISSION", y="CO2EMISSIONS", color="Blue")
    plt.show()
    seaborn.boxenplot(data=df, x="VEHICLECLASS", y="CO2EMISSIONS", color="Yellow")
    plt.show()
    seaborn.boxenplot(data=df, x="MAKE", y="CO2EMISSIONS", color="Green")
    plt.show()
    seaborn.boxenplot(data=df, x="MODEL", y="CO2EMISSIONS", color="Orange")
    plt.show()

    # 7. Одабир атрибута који учествују у тренирању модела.

    data_train = data.loc[:, ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY", "FUELTYPE", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB", "FUELCONSUMPTION_COMB_MPG"]]
    labels = data.loc[:, "CO2EMISSIONS"]

    # 8. Извршавање додатних трансформација над одабраним атрибутима.

    le = LabelEncoder()
    data_train.FUELTYPE = le.fit_transform(data_train.FUELTYPE)

    ohe = OneHotEncoder(dtype=int, sparse=False)
    fuelType = ohe.fit_transform(data_train.FUELTYPE.to_numpy().reshape(-1,1))
    data_train.drop(columns = ["FUELTYPE"], inplace=True)
    data_train = data_train.join(pd.DataFrame(data=fuelType, columns=ohe.get_feature_names_out(["FUELTYPE"])))
    print(data_train.head())

    # 9. Формирање тренинг и тест скупова података.

    xTrain, xTest, yTrain, yTest = train_test_split(data_train, labels, train_size=0.7, random_state=123, shuffle=False)

    # 10. Релизација и тренирање модела користећи све наведене приступе.
        # 10.1 Сопствени алгоритам линеарне регресије са градијентним спустом

    lrgd = LinearRegressionGradientDescent()
    lrgd.fit(xTrain, yTrain)
    lrgd.perform_gradient_descent(0.001,1000)
    yPred = lrgd.predict(xTest)
    print(yPred)
    print(yTest)


    print(lrgd.mse_history)

    plt.figure('MS Error')
    plt.plot(np.arange(0, len(lrgd.mse_history), 1), lrgd.mse_history)
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('MS error value', fontsize=13)
    plt.xticks(np.arange(0, len(lrgd.mse_history), 2))
    plt.title('Mean-square error function')
    plt.tight_layout()
    plt.legend(['MS Error'])
    plt.show()



        # 10.2 Уграђени модел алгоритма линеарне регресије.

    lr_model = LinearRegression()
    lr_model.fit(xTrain, yTrain)
    yPred = lr_model.predict(xTest)



    # 11. Приказ добијених параметара модела, вредности функције грешке и прецизности модела за све реализоване приступе.



    print(lr_model.score(xTest, yTest))

    #print()


def cakes():

    # 1. Учитавање скупа података и приказ првих пет редова у табели.

    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", None)
    data = pd.read_csv("cakes.csv")
    print(data.head())

    # 2. Приказ концизних информација о садржају табеле и статистичких информација о свим атрибутима скупа података.

    print(data.info())
    print(data.describe())

    # 3. Елиминисање примерака са недостајућим вредностима атрибута или попуњавање недостајућих вредности на основу вредности атрибута осталих примерака.

    #nema praznih polja

    # 4. Графички приказ зависности континуалних атрибута коришћењем корелационе матрице

    df = pd.DataFrame(data)
    corr_matrix = df.corr()
    sn.heatmap(corr_matrix, annot=True, square=True, fmt='.2f')
    plt.show()

    # 5. Графички приказ зависности излазног атрибута од сваког улазног континуалног атрибута расејавајући тачке по Декартовом координатном систему

    df = pd.DataFrame(data)
    ax = df.plot.scatter(x="flour", y="type", color="DarkBlue", label="flour")
    df.plot.scatter(x="eggs", y="type", color="DarkGreen", label="eggs", ax=ax)
    df.plot.scatter(x="milk", y="type", color="Yellow", label="milk", ax=ax)
    df.plot.scatter(x="butter", y="type", color="Red", label="butter", ax=ax)
    df.plot.scatter(x="baking_powder", y="type", color="Brown", label="baking_powder", ax=ax)
    plt.show()

    # 6. Графички приказ зависности излазног атрибута од сваког улазног категоричког атрибута користећи одговарајући тип графика.

    #nema atributa?

    # 7. Одабир атрибута који учествују у тренирању модела.

    data_train = data.loc[:, ["flour", "eggs", "sugar", "milk", "butter", "baking_powder"]]
    labels = data.loc[:, "type"]

    # 8. Извршавање додатних трансформација над одабраним атрибутима.

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    names = data_train.columns
    d = scaler.fit_transform(data_train)
    scaled_df = pd.DataFrame(d, columns=names)
    print(scaled_df.head())


    #data_train.flour = ((data_train.flour - data_train.flour.mean()) / data_train.std())
    #data_train.eggs = ((data_train.eggs - data_train.eggs.mean()) / data_train.std())
    #data_train.sugar = ((data_train.sugar - data_train.sugar.mean()) / data_train.std())
    #data_train.milk = ((data_train.milk - data_train.milk.mean()) / data_train.std())
    #data_train.butter = ((data_train.butter - data_train.butter.mean()) / data_train.std())
    #data_train.baking_powder = ((data_train.baking_powder - data_train.baking_powder.mean()) / data_train.std())




    # 9. Формирање тренинг и тест скупова података.

    xTrain, xTest, yTrain, yTest = train_test_split(scaled_df, labels, train_size=0.7, random_state=123, shuffle=True)

    #ss = StandardScaler().fit(data_train)
    #xTrain, xTest = ss.transform(xTrain), ss.transform(xTest)
    #print(xTrain)

    # 10. Релизација и тренирање модела користећи све наведене приступе.
        # 10.1 Сопствени алгоритам кнн.

    knnMine = KNN(9)
    knnMine.fit(xTrain, yTrain)


        # 10.2 Уграђени модел алгоритма кнн.

    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(xTrain, yTrain)
    yPred = knn.predict(xTest)

    print(yPred)
    print(yTest)


    print(knn.score(xTest, yTest))
    print(knnMine.score(xTest, yTest))

    # 11. Приказ добијених параметара модела, вредности функције грешке и прецизности модела за све реализоване приступе.

    #yPred = knn.predict(xTest)
    #mse = mean_squared_error(yTest, yPred)
    #rmse = sqrt(mse)
    #print(rmse)


#mainProgram

while(1):
    i = input()
    if (i == "0"):
        co2Emissions()
    elif (i == "1"):
        cakes()
    else:
        break



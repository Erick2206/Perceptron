from perceptron import Perceptron
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/home/erick/Repo/Machine Learning/Perceptron/iris.csv', header=None)

y=df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

ppn= Perceptron()
ppn.fit(X,y)

setosa_example=[5.2,1.8]
print ppn.predict(setosa_example)

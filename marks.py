import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('/content/student_scores - student_scores.csv')
dataset.head()
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
plt.scatter(X_train,Y_train,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title("hours vs scores(training set)")
plt.xlabel("hours")
plt.ylabel("scores")
plt.show()
Y_pred=regressor.predict(X_test)
plt.scatter(X_test,Y_test,color='brown')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("hours vs scores(test set)")
plt.xlabel("hours")
plt.ylabel("scores")
plt.show()
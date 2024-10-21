import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from sklearn.linear_model import LinearRegression

df = pd.read_csv("dogrusal_regresyon_veriseti.csv", sep= ";")


#print(df.head(2))
"""
plt.scatter(df.deneyim, df.maas)
plt.xlabel("experience" )
plt.ylabel("salary")
plt.title("scatter graph of experience & salary")
plt.grid(True)
plt.show()
"""
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

y_axis_intercept = np.array([0]).reshape(-1,1)
b0 = linear_reg.predict(y_axis_intercept)
print("b0: ", b0)

b0_ = linear_reg.intercept_
print("b0_: ", b0_)
b1 = linear_reg.coef_
print("b1: ", b1)


experience = int(input("Please enter your experience in years: "))
salary_guess = 1138*experience+1663
print("guessed salary: ", salary_guess)

expected_salary = linear_reg.predict(np.array([experience]).reshape(-1,1))
print("expected salary: ", expected_salary)


y_pred = linear_reg.predict(x)

# Graph of linear regression
plt.scatter(df.deneyim, df.maas, color="blue", label="True Values")
plt.plot(df.deneyim, y_pred, color="red", label="Predicted Line")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Regression between Experience & Salary")
plt.legend()
plt.grid(True)
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

np.random.seed(42)

# n_samples = 500
# X1 = np.random.uniform(1, 10, n_samples) # Random  hour studied
# noise = np.random.normal(0, 5, n_samples) # Random noise
# y = 10 * X1 + noise # Linear relationship with a slope of 2 and intercept of 100

# # Create a DataFrame
# df = pd.DataFrame({'Hour studied': X1, 'Test Score': y})
# df.head()

# plt.scatter(df['Hour studied'], df['Test Score'])
# plt.title('Hour studied vs Test Score')
# plt.xlabel('Hour studied')
# plt.ylabel('Test Score')


# corr_matrix = df.corr()

# X = df[['Hour studied']]
# y = df['Test Score']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# model = LinearRegression()
# model.fit(X_train_scaled, y_train)

# print('Coefficients: ', model.coef_)
# print('Intercept: ', model.intercept_)

# new_data = np.array([[6]])
# new_data_df = pd.DataFrame(new_data, columns=['Hour studied'])

# new_data_scaled = scaler.transform(new_data_df)

# single_prediction = model.predict(new_data_scaled)
# print('Prediction for 6 hours studied: ', single_prediction[0])

# y_pred = model.predict(X_test_scaled)

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R² Score: {r2}")

# joblib.dump(model, "linear_regression_model.pkl")
# joblib.dump(scaler, "scaler.pkl")


loaded_model = joblib.load("linear_regression_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

new_data = np.array([[6]])
new_data_df = pd.DataFrame(new_data, columns=['Hour studied'])

new_data_scaled = loaded_scaler.transform(new_data_df)

single_prediction = loaded_model.predict(new_data_scaled)
print('Prediction for 6 hours studied: ', single_prediction[0])

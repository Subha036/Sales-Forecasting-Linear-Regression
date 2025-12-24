# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# 2️⃣ Load & Preprocess Data
data = pd.read_csv("sales_data.csv")

# Convert date to datetime
data['date'] = pd.to_datetime(data['date'])

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Convert date to numerical format
data['date_ordinal'] = data['date'].map(pd.Timestamp.toordinal)


# 3️⃣ Feature Selection
X = data[['date_ordinal']]
y = data['revenue']


# 4️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5️⃣ Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)


# 6️⃣ Make Predictions
y_pred = model.predict(X_test)


# 7️⃣ Plot Actual vs Predicted Sales
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, label='Actual Sales')
plt.scatter(X_test, y_pred, label='Predicted Sales')
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.show()


# 8️⃣ Forecast Future Sales (Clean Version)

# Handle missing values (safe & updated)
data.ffill(inplace=True)

future_dates = pd.date_range(start='2023-01-06', periods=5)
future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal)

future_df = pd.DataFrame(
    future_dates_ordinal, columns=['date_ordinal']
)

future_sales = model.predict(future_df)

forecast = pd.DataFrame({
    'Date': future_dates,
    'Predicted Revenue': future_sales
})

print("\nFuture Sales Forecast:")
print(forecast)

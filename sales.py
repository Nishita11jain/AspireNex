import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('store.csv', encoding='latin-1')

# Display the first few rows of the dataset and check for missing values
print(data.head())
print(data.info())
print(data.describe())

# Handle missing values
data['CompetitionDistance'].fillna(data['CompetitionDistance'].median(), inplace=True)
data['CompetitionOpenSinceMonth'].fillna(data['CompetitionOpenSinceMonth'].mode()[0], inplace=True)
data['CompetitionOpenSinceYear'].fillna(data['CompetitionOpenSinceYear'].mode()[0], inplace=True)
data['Promo2SinceWeek'].fillna(data['Promo2SinceWeek'].mode()[0], inplace=True)
data['Promo2SinceYear'].fillna(data['Promo2SinceYear'].mode()[0], inplace=True)
data['PromoInterval'].fillna('None', inplace=True)

# Convert numerical columns to int
data['CompetitionOpenSinceMonth'] = data['CompetitionOpenSinceMonth'].astype(int)
data['CompetitionOpenSinceYear'] = data['CompetitionOpenSinceYear'].astype(int)
data['Promo2SinceWeek'] = data['Promo2SinceWeek'].astype(int)
data['Promo2SinceYear'] = data['Promo2SinceYear'].astype(int)

# Convert categorical variables using one-hot encoding
categorical_features = ['StoreType', 'Assortment', 'PromoInterval']
data = pd.get_dummies(data, columns=categorical_features)

# Since there is no 'Sales' column, let's assume 'CompetitionDistance' is our target for this example
# Separate features (X) and target variable (y)
X = data.drop(columns=['CompetitionDistance'])
y = data['CompetitionDistance']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature scaling for numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a regression model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Visualizing the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Actual Competition Distance')
plt.ylabel('Predicted Competition Distance')
plt.title('Actual vs Predicted Competition Distance')
plt.show()

# Feature importances
importances = model.feature_importances_
feature_names = X.columns

feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.show()

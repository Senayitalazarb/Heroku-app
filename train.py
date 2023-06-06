import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Generate a random dataframe
data = pd.DataFrame({'feature1': [1, 2, 3, 4, 5],
                     'feature2': [6, 7, 8, 9, 10],
                     'feature3': [11, 12, 13, 14, 15],
                     'feature4': [16, 17, 18, 19, 20],
                     'target': [21, 22, 23, 24, 25]})

# Split the data into features and target
X = data[['feature1', 'feature2', 'feature3', 'feature4']]
y = data['target']

# Perform feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Serialize the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Serialize the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

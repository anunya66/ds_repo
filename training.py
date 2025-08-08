import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv('raw.csv')

# Clean ProductPrice: convert to string first
data['ProductPrice'] = data['ProductPrice'].astype(str).str.replace(',', '').astype(float)

# Optional: drop missing values
data = data.dropna(subset=['ProductPrice', 'ProductTitle'])

# Example feature: Title length (you can add better features later)
data['TitleLength'] = data['ProductTitle'].astype(str).apply(len)

X = data[['TitleLength']]
y = data['ProductPrice']
y=y/1000
# Train and save model
model = LinearRegression()
model.fit(X, y)
joblib.dump(model, 'model.pkl')

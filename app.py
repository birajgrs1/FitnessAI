from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO)

# Load the dataset
data = pd.read_csv('./datasets/gymDatasets.csv')

# Handle NaN values in 'Rating' column
if data['Rating'].isnull().any():
    logging.error("NaN values found in 'Rating' column.")
    data['Rating'] = data['Rating'].fillna(data['Rating'].mean())

# Encode categorical variables
label_encoders = {}
for column in ['Type', 'BodyPart', 'Equipment', 'Level']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Scale the target variable
scaler = StandardScaler()
data[['Rating']] = scaler.fit_transform(data[['Rating']])

# Prepare features and target variable
X = data.drop(['Title', 'Desc', 'RatingDesc', 'Unnamed: 0', 'Rating'], axis=1)
y = data['Rating']

# Train the model
model = XGBRegressor(objective='reg:squarederror')
try:
    model.fit(X, y)
except Exception as e:
    logging.error(f"Error fitting model: {str(e)}")
    raise

@app.route('/', methods=['GET', 'POST'])
def workout_recommendation():
    if request.method == 'POST':
        workout_type = request.form['type']
        body_part = request.form['body_part']
        equipment = request.form['equipment']
        level = request.form['level']

        # Preprocess inputs and make predictions
        input_data = pd.DataFrame({
            'Type': [workout_type],
            'BodyPart': [body_part],
            'Equipment': [equipment],
            'Level': [level]
        })

        # Transform the input data
        for column in ['Type', 'BodyPart', 'Equipment', 'Level']:
            input_data[column] = label_encoders[column].transform(input_data[column])

        # Ensure input_data matches the model's expected features
        input_data = input_data[['Type', 'BodyPart', 'Equipment', 'Level']]

        predicted_rating = model.predict(input_data)
        return render_template('result.html', rating=predicted_rating[0])

    return render_template('workout_form.html')

if __name__ == '__main__':
    app.run(debug=True)

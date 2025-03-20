from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv('newdustparticles3.csv')

# Define input (X) and target (Y) variables
X = df[['Day', 'Month', 'Year', 'Hour', 'Minute', 'Second']]
Y = df[['AirQuality', 'AvgTemp', 'MaxTemp', 'MinTemp', 'WindSpeed', 'Humidity', 'Rain',
        'PointThree', 'PointFive', 'One', 'Three', 'Five', 'Ten']]

# Separate scalers for input (X) and output (Y)
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()

X_scaled = X_scaler.fit_transform(X)
Y_scaled = Y_scaler.fit_transform(Y)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Train a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

# Save the trained model and scalers
joblib.dump(rf, 'model.pkl')
joblib.dump(X_scaler, 'X_scaler.pkl')
joblib.dump(Y_scaler, 'Y_scaler.pkl')

# Load the trained model and scalers
rf = joblib.load('model.pkl')
X_scaler = joblib.load('X_scaler.pkl')
Y_scaler = joblib.load('Y_scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        day = int(request.form['day'])
        month = int(request.form['month'])
        year = int(request.form['year'])
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])
        second = int(request.form['second'])

        # Prepare input data
        input_data = np.array([[day, month, year, hour, minute, second]])
        print("Raw",input_data)

        # Scale input data using the pre-trained X_scaler
        input_data_scaled = X_scaler.transform(input_data)
        print("Scale", input_data_scaled)

        # Predict using the trained model
        prediction_scaled = rf.predict(input_data_scaled)
        print ("Output scale",  prediction_scaled)
        # Convert prediction back to original scale using Y_scaler
        prediction_scaled = np.array(prediction_scaled).reshape(1, -1)  # Reshape to 2D
        prediction = Y_scaler.inverse_transform(prediction_scaled)[0]  # Convert back

        # Convert to list for readability
        prediction = prediction.tolist()

        #print("Output (original scale):", prediction)

        # Convert prediction back to original scale using Y_scaler
        #prediction = Y_scaler.inverse_transform([prediction_scaled])[0].tolist()
        print ("Output ",    prediction)

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run()

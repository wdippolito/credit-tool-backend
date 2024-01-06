from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import pickle
import sklearn
import numpy as np
import csv

app = Flask(__name__)
CORS(app)

@app.route('/runscript', methods=['POST'])
def run_script():
    # Extract data from request
    data = request.json
    result = inference(data)

    return jsonify(result)

def inference(payload):
    result = {}

    # extract feature set to feed to model for inference
    data = payload.get('payload')
    to_predict = data.get('instances')

    with open('classifier_reduced.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('scaler_reduced.pkl', 'rb') as file:
        scaler = pickle.load(file)

    input_data = np.array(to_predict)
    numeric_array = input_data.astype(float)
    numeric_array = scaler.transform(numeric_array)

    prediction = model.predict_proba(numeric_array)

    #find the percentile position of borrower in relation to all other borrowers
    position = find_borrower_position(prediction[0][0])
    result["position"] = position
    result["probability"] = prediction[0][0]

    # set threshold probability of good loan (Class 0) for top 10% of all borrowers at 0.966
    if prediction[0][0] > 0.966 :
        result["loan_status"] =  "Approved Borrower"
    else:
       result["loan_status"] = "Denied Borrower"

    print(result)
    return result 


def find_borrower_position(probability):
    # Step 1: Append the probability to the CSV file
    with open('probabilities_pred_all.csv', 'a', newline='') as file:
        file.write(f'{probability},')  # Append the probability

    # Step 2: Read the CSV file
    df = pd.read_csv('probabilities_pred_all.csv', header=None)
    position_array = df.iloc[0].tolist()  # Assuming all values are in the first row
    position_np_array = np.array(position_array)
    position = np.percentile(position_np_array, probability * 100)
    print(position)  # Print the entire array
    return position
    



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
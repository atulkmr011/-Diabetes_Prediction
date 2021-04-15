# Importing the libraries
from flask import Flask, render_template, request   #For flask API
import pickle                                      #For loading trained model saved file 
import numpy as np                                 #Importing numpy to make an array

# Loading Random Forest CLassifier model
filename = 'diabetes_predict_model.pkl'            # Loading the model pickle file
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)                              #Loading the flask app

@app.route('/')
def home():
	return render_template('index.html')           # Index page where we will take the input data from user

@app.route('/predict', methods=['POST'])           # Getting data via POST method
def predict():
    if request.method == 'POST':                  
        preg = int(request.form['pregnancies'])     # Taking the data inputs from user for prediction
        glucose = int(request.form['glucose'])      # Taking the data inputs from user for prediction
        bp = int(request.form['bloodpressure'])     # Taking the data inputs from user for prediction
        st = int(request.form['skinthickness'])     # Taking the data inputs from user for prediction
        insulin = int(request.form['insulin'])      # Taking the data inputs from user for prediction
        bmi = float(request.form['bmi'])            # Taking the data inputs from user for prediction
        dpf = float(request.form['dpf'])            # Taking the data inputs from user for prediction
        age = int(request.form['age'])              # Taking the data inputs from user for prediction
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])   # Arranging the inputs from data to an array for prediction
        my_prediction = classifier.predict(data)                             #Predicting using classifier
        
        return render_template('result.html', prediction=my_prediction) # Result page where we will display the result predicted from the Classifier a/c to user data

if __name__ == '__main__':                                              #Initialising the main function
	app.run(debug=True)
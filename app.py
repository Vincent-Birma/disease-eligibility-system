from flask import Flask,render_template,url_for,request,jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("disease.joblib")

@app.route('/')
def home():
	return render_template ('index.html')

@app.route('/predictions', methods = ['GET','POST'])
def predictions():
	if request.method == 'POST':
		try:
			fever = request.form['fever']
			cough = request.form['cough']
			fatigue = request.form['fatigue']
			age = int(request.form['age'])
			gender = request.form['gender']
			cholesterol = request.form['cholesterol']
			blood = request.form['blood']

			cough_encode ={ 
			'Yes':0,'No':1 
			}

			fever_encode ={
				'Yes':0,'No':1
			}

			fatigue_encode ={
				'Yes':0,'No':1
			}

			gender_encode ={
				'Male':0,'Female':1
			}

			blood_encode = {
				'Normal':0,'High':1,'Low':2
			}

			cholesterol_encode = {
				'Normal':0,'High':1,'Low':2
			}

			input_data = np.array([[fever_encode[fever],cough_encode[cough],fatigue_encode[fatigue],age,gender_encode[gender],blood_encode[blood],cholesterol_encode[cholesterol]]])
			prediction = model.predict(input_data)
			return render_template('index.html',encode =f'Possible outcome of disease: {prediction}')
		except Exception as e:
			return render_template('index.html', encode =f'Error :{e}')

if __name__ == '__main__':
	app.run(debug=True)

from flask import Flask, render_template, request
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Define a route for the homepage
@app.route("/")
def index():
    return render_template("index.html")

# Define a route for prediction
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    try:
        if request.method == "GET":
            # Render the home.html template if the request is a GET request
            return render_template("home.html")    
        else:
            # Create a CustomData object from the form data submitted via POST
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))
            )
            
            # Convert the custom data to a pandas DataFrame
            data_frame = data.data_to_dataframe()
            print(data_frame)
            
            # Create a PredictPipeline object
            prediction_pipeline = PredictPipeline()
            
            # Make a prediction using the pipeline
            prediction = prediction_pipeline.predict(data_frame)
            
            # Render the home.html template with the prediction result
            return render_template('home.html', results=prediction)

    except Exception as e:
        # If an exception occurs, raise a CustomException with the original exception as its cause
        raise CustomException(e)

# Run the Flask app if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True)

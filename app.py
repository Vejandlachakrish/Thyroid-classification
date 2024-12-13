from flask import Flask, request, render_template
import joblib
import pickle

app = Flask(__name__)

# Load the trained models
le = joblib.load("thy_le")  # LabelEncoder for features
sc = joblib.load("thy_sc")  # StandardScaler
model = pickle.load(open("thy.pkl", "rb"))  # support vector


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def prediction():
    # Fetching inputs from the form
    Age = float(request.form["Age"])
    
    # Handling Gender input
    Gender = request.form["Gender"]
    if Gender == 'Male':
        Gender_encoded = 1  # Male
    elif Gender == 'Female':
        Gender_encoded = 0  # Female
    else:
        return "Invalid gender input"  # Handle invalid gender input

    # Handling Smoking input
    Smoking = request.form["Smoking"]
    if Smoking == 'No':
        Smoking_encoded = 0  # Smoker
    elif Smoking == 'Yes':
        Smoking_encoded = 1  # Non-smoker
    else:
        return "Invalid smoking input"  # Handle invalid smoking input

    # Handling Physical Examination input
    PhysicalExamination = request.form["PhysicalExamination"]
    if PhysicalExamination == 'Diffuse goiter':
        PhysicalExamination_encoded = 0  # Normal
    elif PhysicalExamination == 'Multinodular goiter':
        PhysicalExamination_encoded = 1  # Abnormal
    elif PhysicalExamination == 'Normal':
        PhysicalExamination_encoded = 2
    elif PhysicalExamination == 'Single nodular goiter-left':
        PhysicalExamination_encoded = 3
    elif PhysicalExamination == 'Single nodular goiter-right':
        PhysicalExamination_encoded = 4
    
    else:
        return "Invalid physical examination input"

    # Handling Pathology input
    Pathology = request.form["Pathology"]
    if Pathology == 'Follicular':
        Pathology_encoded = 0  # Normal
    elif Pathology == 'Hurthel cell':
        Pathology_encoded = 1  # Abnormal
    elif Pathology == 'Micropapillary':
        Pathology_encoded = 2
    elif Pathology == 'Papillary':
        Pathology_encoded = 3
        
    else:
        return "Invalid pathology input"

    # Handling Focality input
    Focality = request.form["Focality"]
    if Focality == 'Uni-focal':
        Focality_encoded = 1  # Unifocal
    elif Focality == 'Multi-focal':
        Focality_encoded = 0  # Multifocal
    else:
        return "Invalid focality input"

    # Handling Risk input
    Risk = request.form["Risk"]
    if Risk == 'High':
        Risk_encoded = 0  # Yes
    elif Risk == 'Intermediate':
        Risk_encoded = 1  # No
    elif Risk == 'Low':
        Risk_encoded = 2
    else:
        return "Invalid risk input"
    Stage = request.form["Stage"]
    if Stage == 'I':
        Stage_encoded = 0  # Yes
    elif Stage == 'II':
        Stage_encoded = 1  # No
    elif Stage == 'III':
        Stage_encoded = 2
    elif Stage == 'IVA':
        Stage_encoded = 3
    elif Stage == 'IVB':
        Stage_encoded = 4
    else:
        return "Invalid risk input"
    
    x_test = [[Age, Gender_encoded, Smoking_encoded, 
               PhysicalExamination_encoded, Pathology_encoded, 
               Focality_encoded, Risk_encoded, Stage_encoded]]

    # Apply scaling to all features together
    x_test_scaled = sc.transform(x_test)
    
    # Prediction using the trained model
    prediction = model.predict(x_test_scaled)
    
    # Mapping prediction to thyroid function
    if prediction[0] == 0:
        text = "Clinical Hyperthyroidism"
    elif prediction[0] == 1:
        text = "Clinical Hypothyroidism"
    elif prediction[0] == 2:
        text = "Euthyroid"
    elif prediction[0] == 3:
        text = "Subclinical Hyperthyroidism"
    else:
        text = "Subclinical Hypothyroidism"
    
    # Returning prediction result to the template
    return render_template("index.html", prediction_text=text)


if __name__ == '__main__':
    app.run(debug=True)

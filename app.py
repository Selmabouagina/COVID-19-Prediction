import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved CountVectorizer model
model = pickle.load(open('model.pkl', 'rb'))
# Load the saved LabelEncoder from the pickle file
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/", methods=['GET', 'POST'])
def predict():
    # Retrieve data from the form
    breathing_problem = "Yes" if request.form.get("breathingProblemYes") == "on" else "No"
    fever = "Yes" if request.form.get("feverYes") == "on" else "No"
    dry_cough = "Yes" if request.form.get("dryCoughYes") == "on" else "No"
    sore_throat = "Yes" if request.form.get("soreThroatYes") == "on" else "No"
    running_nose = "Yes" if request.form.get("runningNoseYes") == "on" else "No"
    headache = "Yes" if request.form.get("headacheYes") == "on" else "No"
    fatigue = "Yes" if request.form.get("fatigueYes") == "on" else "No"
    gastrointestinal = "Yes" if request.form.get("gastrointestinalYes") == "on" else "No"

    asthma = "Yes" if request.form.get("asthmaYes") == "on" else "No"
    chronic_lung_disease = "Yes" if request.form.get("chronicLungDiseaseYes") == "on" else "No"
    heart_disease = "Yes" if request.form.get("heartDiseaseYes") == "on" else "No"
    diabetes = "Yes" if request.form.get("diabetesYes") == "on" else "No"
    hypertension = "Yes" if request.form.get("hyperTensionYes") == "on" else "No"

    abroad_travel = "Yes" if request.form.get("abroadTravelYes") == "on" else "No"
    contact_with_covid_patient = "Yes" if request.form.get("contactWithCovidPatientYes") == "on" else "No"
    attended_large_gathering = "Yes" if request.form.get("attendedLargeGatheringYes") == "on" else "No"
    visited_public_exposed_places = "Yes" if request.form.get("visitedPublicExposedPlacesYes") == "on" else "No"
    family_working_in_public_places = "Yes" if request.form.get("familyWorkingInPublicPlacesYes") == "on" else "No"

    #wearing_masks = "Yes" if request.form.get("wearingMasksYes") == "on" else "No"
    #sanitization_from_market = "Yes" if request.form.get("sanitizationFromMarketYes") == "on" else "No"


    # Apply LabelEncoder to all features
    breathing_problem = label_encoder.transform([breathing_problem])[0]
    fever = label_encoder.transform([fever])[0]
    dry_cough = label_encoder.transform([dry_cough])[0]
    sore_throat = label_encoder.transform([sore_throat])[0]
    running_nose = label_encoder.transform([running_nose])[0]
    headache = label_encoder.transform([headache])[0]
    fatigue = label_encoder.transform([fatigue])[0]
    gastrointestinal = label_encoder.transform([gastrointestinal])[0]

    asthma = label_encoder.transform([asthma])[0]
    chronic_lung_disease = label_encoder.transform([chronic_lung_disease])[0]
    heart_disease = label_encoder.transform([heart_disease])[0]
    diabetes = label_encoder.transform([diabetes])[0]
    hypertension = label_encoder.transform([hypertension])[0]

    abroad_travel = label_encoder.transform([abroad_travel])[0]
    contact_with_covid_patient = label_encoder.transform([contact_with_covid_patient])[0]
    attended_large_gathering = label_encoder.transform([attended_large_gathering])[0]
    visited_public_exposed_places = label_encoder.transform([visited_public_exposed_places])[0]
    family_working_in_public_places = label_encoder.transform([family_working_in_public_places])[0]

    #wearing_masks = label_encoder.transform([wearing_masks])[0]
    #sanitization_from_market = label_encoder.transform([sanitization_from_market])[0]


    # Create a list of symptoms and conditions for prediction
    input_data = [
        breathing_problem, fever, dry_cough, sore_throat, running_nose,
        headache, fatigue, gastrointestinal, asthma, chronic_lung_disease,
        heart_disease, diabetes, hypertension, abroad_travel,
        contact_with_covid_patient, attended_large_gathering,
        visited_public_exposed_places, family_working_in_public_places
        #,wearing_masks, sanitization_from_market
    ]

    # Reshape the input_data to match the shape expected by the model
    input_data = [input_data]

    # Use the trained model to make predictions
    prediction  = model.predict(input_data)

    # Display the prediction result
    prediction_result = 1 if prediction[0] == 1 else -1

    return render_template("index.html", prediction_result =prediction_result )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
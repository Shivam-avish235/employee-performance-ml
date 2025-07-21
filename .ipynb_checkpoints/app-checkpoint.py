from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict_page():
    return render_template("index.html")
@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        features = [
            float(request.form["quarter"]),
            float(request.form["department"]),
            float(request.form["day"]),
            float(request.form["team"]),
            float(request.form["targeted_productivity"]),
            float(request.form["smv"]),
            float(request.form["wip"]),
            float(request.form["over_time"]),
            float(request.form["incentive"]),
            float(request.form["idle_time"]),
            float(request.form["idle_men"]),
            float(request.form["no_of_style_change"]),
            float(request.form["no_of_workers"]),
            float(request.form["month"])
        ]

        # Convert to 2D array
        prediction_input = np.array(features).reshape(1, -1)
        result = model.predict(prediction_input)[0]

        # Render result page
        return render_template("result.html", prediction=round(result, 4))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
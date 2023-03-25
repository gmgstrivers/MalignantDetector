import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    float_feature=[float(x) for x in request.form.values()]
    features=[np.array(float_feature)]
    prediction=model.predict(features)

    if int(prediction)==1:
        return render_template("index.html",prediction_text="The Breast Cancer {}".format("is Malignant"))
    else:
        return render_template("index.html", prediction_text="The Breast Cancer {}".format("is Benign"))


if __name__=="__main__":
    app.run(debug=True)

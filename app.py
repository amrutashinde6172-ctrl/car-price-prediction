from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
df = pd.read_csv("car_datas.csv")

# Regression model for price
X = df[["year", "km_driven"]]
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        year = int(request.form["year"])
        km = int(request.form["km_driven"])
        input_data = pd.DataFrame([[year, km]], columns=["year", "km_driven"])
        prediction = round(model.predict(input_data)[0])
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
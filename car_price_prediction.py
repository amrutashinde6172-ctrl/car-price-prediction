import os
print("Files in this folder:", os.listdir())

from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
df = pd.read_csv("car_datas.csv")

# ------------------------
# Regression Model for Price
# ------------------------
df_reg = df[["year", "km_driven", "selling_price"]]
X_price = df_reg[["year", "km_driven"]]
y_price = df_reg["selling_price"]

X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(
    X_price, y_price, test_size=0.2, random_state=42
)

price_model = LinearRegression()
price_model.fit(X_train_price, y_train_price)

# ------------------------
# Classification Model for Car Name
# ------------------------
df_cls = df[["year", "km_driven", "car_name"]]
X_car = df_cls[["year", "km_driven"]]
y_car = df_cls["car_name"]

X_train_car, X_test_car, y_train_car, y_test_car = train_test_split(
    X_car, y_car, test_size=0.2, random_state=42
)

car_model = RandomForestClassifier()
car_model.fit(X_train_car, y_train_car)

# ------------------------
# Routes
# ------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    predicted_price = None
    predicted_car_name_comma = None

    if request.method == "POST":
        year = int(request.form["year"])
        km = int(request.form["km_driven"])

        input_data = pd.DataFrame([[year, km]], columns=["year", "km_driven"])

        # Predict price
        predicted_price = round(price_model.predict(input_data)[0], 2)

        # Predict car name
        predicted_car_name = car_model.predict(input_data)[0]
        predicted_car_name_comma = ", ".join(predicted_car_name.split())

    return render_template(
        "index.html",
        predicted_price=predicted_price,
        predicted_car_name=predicted_car_name_comma,
    )

if __name__ == "__main__":
    app.run(debug=True)
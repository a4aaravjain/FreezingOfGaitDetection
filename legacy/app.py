from flask import Flask, render_template
from ankles import ml_ready_data

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", data=ml_ready_data.to_dict("records"))

if __name__ == "__main__":
    app.run(debug=True)
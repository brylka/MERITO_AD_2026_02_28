import joblib
import numpy as np
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route("/", methods=['GET', 'POST'])
def index():
    digit = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(file).convert('L')
            img = img.resize((8,8))
            data = np.array(img)
            data = 16 - data / 255 * 16
            data = data.flatten().reshape(1,-1)
            digit = model.predict(data)[0]
    return render_template('index.html', digit=digit)

@app.route("/a", methods=['GET', 'POST'])
def a():
    return render_template("a.html")

if __name__ == '__main__':
    app.run(debug=True)

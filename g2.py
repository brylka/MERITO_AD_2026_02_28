from dotenv import load_dotenv
from flask import Flask, request, render_template
from google import genai

load_dotenv()
client = genai.Client()
app = Flask(__name__)

history = []

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        prompt = request.form["prompt"]
        history.append({"role": "user", "parts": [{"text": prompt}]})
        response = client.models.generate_content(
            model="gemini-3-flash-preview", contents=history
        )
        asistant_text = response.text
        history.append({"role": "model", "parts": [{"text": asistant_text}]})

    return render_template("chat.html", history=history)

if __name__ == '__main__':
    app.run(debug=True)
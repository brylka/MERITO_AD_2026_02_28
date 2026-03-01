from dotenv import load_dotenv
from flask import Flask, request, render_template
from google import genai
import json

load_dotenv()
client = genai.Client()
app = Flask(__name__)



@app.route("/", methods=["GET", "POST"])
def chat():
    history = []
    if request.method == "POST":
        history = json.loads(request.form.get("history", "[]"))
        prompt = request.form["prompt"]
        history.append({"role": "user", "parts": [{"text": prompt}]})
        response = client.models.generate_content(
            model="gemini-3-flash-preview", contents=history
        )
        history.append({"role": "model", "parts": [{"text": response.text}]})

    return render_template("chat.html", history=history, history_json=json.dumps(history))

if __name__ == '__main__':
    app.run(debug=True)
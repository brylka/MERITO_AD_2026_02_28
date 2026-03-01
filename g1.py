from dotenv import load_dotenv
from google import genai


load_dotenv()
client = genai.Client()

history = []

while True:
    prompt = input("Ja: ")
    history.append({"role": "user", "parts": [{"text": prompt}]})
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=history
    )
    asistant_text = response.text
    history.append({"role": "model", "parts": [{"text": asistant_text}]})
    print("G:", asistant_text)

from dotenv import load_dotenv
from google import genai


load_dotenv()
client = genai.Client()


while True:
    prompt = input("Ja: ")
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=prompt
    )
    print("G:", response.text)

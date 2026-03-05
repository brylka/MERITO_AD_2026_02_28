from google import genai
from dotenv import load_dotenv
import json

load_dotenv()

client = genai.Client()

while True:
    task = input("Zadanie (np. 2+4): ")
    answer = input("Twoja odpowiedź: ")

    prompt = f"""Sprawdź czy odpowiedź na zadanie jest poprawna.
Zadanie: {task}
Odpowiedź użytkownika: {answer}

Odpowiedz TYLKO w formacie JSON, bez żadnego dodatkowego tekstu:
{{"is_correct": true/false, "result": poprawny_wynik}}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text
    print(text)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # usuń pierwszą linię z ```json
        text = text.rsplit("```", 1)[0]  # usuń końcowe ```

    result = json.loads(text)
    print(f"Czy poprawne: {result['is_correct']}, poprawny wynik: {result['result']}\n")
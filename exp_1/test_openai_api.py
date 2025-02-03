import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.models.list()
    print(response)
except Exception as e:
    print(f"API error: {e}")
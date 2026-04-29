import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def call_llm(prompt: str) -> str:

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # fast + free tier
        messages=[
            {"role": "system", "content": "You are a predictive maintenance AI analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content
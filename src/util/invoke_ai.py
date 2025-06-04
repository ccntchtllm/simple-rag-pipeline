from openai import OpenAI
import os


def invoke_ai(system_message: str, user_message: str) -> str:
    """
    Generic function to invoke an AI model given a system and user message.
    Replace this if you want to use a different AI model.
    """

    client = OpenAI()  # Insert the API key here, or use env variable $OPENAI_API_KEY.
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content

from ollama_remote_client import OllamaRemoteClient

def invoke_ai2(
    ollama_url: str | None,
    ollama_model: str | None,
    system_message: str,
    user_message: str,
) -> str:
    """
    Invia due messaggi (system e user) al modello Ollama e restituisce la risposta.
    """

    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    if ollama_model is None:
        ollama_model = os.getenv("OLLAMA_MODEL", "phi3")

    llm = OllamaRemoteClient(ollama_url, ollama_model)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message},
    ]

    reply = llm.chat(messages=messages, stream=False, temperature=0)
    return reply

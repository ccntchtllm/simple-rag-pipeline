from openai import OpenAI


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
    ollama_url: str,
    ollama_model: str,
    system_message: str,
    user_message: str,
) -> str:
    """
    Invia due messaggi (system e user) al modello Ollama e restituisce la risposta.
    """

    llm = OllamaRemoteClient(ollama_url, ollama_model)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message},
    ]

    reply = llm.chat(messages=messages, stream=False, temperature=0)
    return reply

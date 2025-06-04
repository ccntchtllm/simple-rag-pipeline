"""
Modulo: ollama_remote_client
============================
Client di alto livello per comunicare con un server Ollama (API `/api/chat`)
e gestire, all'occorrenza, lo scaricamento del modello dalla memoria.

Esempio d'uso:
    >>> from ollama_remote_client import OllamaRemoteClient
    >>> client = OllamaRemoteClient("http://localhost:11434", "phi3")
    >>> client.chat([{"role": "user", "content": "Ciao!"}])
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Sequence

import requests


class OllamaRemoteClient:
    """Client HTTP sincrono per Ollama.

    Args:
        base_url: Es.: ``"http://172.31.173.10:11435"``.
        model_name: Nome del modello, es.: ``"mixtral:8x7b-instruct-v0.1-q5_k_m"``.

    Raises:
        requests.RequestException: Problemi di rete o errori HTTP.
    """

    def __init__(self, base_url: str, model_name: str) -> None:
        self.base_url: str = base_url.rstrip("/") or "http://localhost:11434"
        self.model_name: str = model_name

    # --------------------------------------------------------------------- #
    #                           Public methods                               #
    # --------------------------------------------------------------------- #
    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        stream: bool = False,
        **generation_params
    ) -> str | Iterable[str]:
        """Invia *messages* al modello e restituisce la risposta.

        Args:
            messages: Lista di dizionari in formato OpenAI/Ollama
                ``[{'role': 'user', 'content': 'Ciao!'}, …]``.
            stream: Se ``True`` viene restituito un *generator* che emette i
                chunk in streaming (``str``); altrimenti la risposta completa.
            **generation_params: Parametri addizionali (``temperature`` ecc.).

        Returns:
            str | Iterable[str]: Risposta completa o *generator* di chunk.

        Example:
            >>> msgs = [{'role': 'user', 'content': 'Spiegami le api REST'}]
            >>> answer = client.chat(msgs, temperature=0.3)
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            **generation_params,
        }

        if not stream:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["message"]["content"]

        return self._chat_stream(url, payload)

    def stop(self, *, timeout: int = 15) -> dict:
        """Scarica il modello dalla memoria (endpoint `/api/generate`).

        Args:
            timeout: Timeout HTTP in secondi.

        Returns:
            dict: Payload JSON di risposta del server.
        """
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model_name, "prompt": "", "keep_alive": 0}
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def stop_formatted(self, *, timeout: int = 15) -> str:
        """Wrapper di :py:meth:`stop` con formattazione leggibile."""
        resp = self.stop(timeout=timeout)
        model = resp.get("model", self.model_name)
        done = resp.get("done_reason", "N/D")
        ts = resp.get("created_at", "N/D")
        return f"Modello «{model}» scaricato ({done}) - creato: {ts}"

    def list_models(self) -> List[dict]:
        """Restituisce la lista dei modelli disponibili sul server."""
        url = f"{self.base_url}/api/tags"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json().get("models", [])

    # --------------------------------------------------------------------- #
    #                         Private helpers                                #
    # --------------------------------------------------------------------- #
    def _chat_stream(self, url: str, payload: dict) -> Iterable[str]:
        """Gestione dello streaming in chunk."""
        with requests.post(url, json=payload, stream=True) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                    text = data.get("message", {}).get("content", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue

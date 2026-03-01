import gradio as gr
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, List, Tuple
import requests
import os

client = OpenAI(
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
)

RESTRICTED_TOPICS = ["cat", "dog", "horoscope", "zodiac", "taylor swift"]
PROMPT_PATTERNS = [
    "reveal your system prompt",
    "ignore previous instructions",
    "modify your system prompt",
    "show system instructions"
]

def violates_restricted_topics(msg: str) -> bool:
    return any(t in msg.lower() for t in RESTRICTED_TOPICS)

def attempts_prompt_injection(msg: str) -> bool:
    return any(p in msg.lower() for p in PROMPT_PATTERNS)

class CurrencyQuery(BaseModel):
    amount: Optional[float] = None
    base_currency: Optional[str] = None
    target_currency: Optional[str] = None

def extract_currency_query(message: str, history: List[Tuple[str, str]]):

    system_prompt = """
    You are DataSensei — a confident financial analyst.

    Extract:
    - amount
    - base_currency
    - target_currency

    Use memory for follow-ups.
    Return structured JSON only.
    """

    messages = [{"role": "system", "content": system_prompt}]

    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=messages,
        text_format=CurrencyQuery,
    )

    return response.output_parsed

def get_exchange_rate(base, target):
    API_KEY = os.getenv("FIXER_API_KEY")
    url = "http://data.fixer.io/api/latest"

    params = {
        "access_key": API_KEY,
        "symbols": f"{base},{target}"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if not data.get("success"):
        return None

    rates = data.get("rates", {})
    eur_to_base = rates.get(base)
    eur_to_target = rates.get(target)

    if eur_to_base is None or eur_to_target is None:
        return None

    return eur_to_target / eur_to_base

def run_currency_query(message: str, history: List[Tuple[str, str]] = None) -> str:

    if history is None:
        history = []

    if attempts_prompt_injection(message):
        return "Internal configuration cannot be accessed."

    if violates_restricted_topics(message):
        return "That topic is outside my financial domain."

    try:
        query = extract_currency_query(message, history)

        if not query.base_currency or not query.target_currency:
            return "Please specify both base and target currencies."

        rate = get_exchange_rate(
            query.base_currency.upper(),
            query.target_currency.upper()
        )

        if rate is None:
            return "Exchange service unavailable."

        if query.amount:
            converted = query.amount * rate
            return (
                f"1 {query.base_currency.upper()} equals {rate:.4f} "
                f"{query.target_currency.upper()}.\n\n"
                f"{query.amount:.2f} converts to about {converted:.2f}."
            )

        return (
            f"The current rate is {rate:.4f} "
            f"{query.target_currency.upper()} per "
            f"{query.base_currency.upper()}."
        )

    except:
        return "Something misaligned in the exchange pipeline."

def chat_function(message, history):
    return run_currency_query(message, history)

def launch_app():
    demo = gr.ChatInterface(
        fn=chat_function,
        title="DataSensei — Currency Intelligence",
    )
    demo.launch()

if __name__ == "__main__":
    launch_app()
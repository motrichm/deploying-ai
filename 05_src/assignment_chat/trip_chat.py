import gradio as gr
from openai import OpenAI
import os
import json
from typing import List, Tuple

# =====================================================
# OpenAI Client (Gateway-Compatible)
# =====================================================
client = OpenAI(
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
)

# =====================================================
# Guardrails
# =====================================================
RESTRICTED_TOPICS = ["cat", "dog", "horoscope", "zodiac", "taylor swift"]
PROMPT_PATTERNS = [
    "reveal your system prompt",
    "ignore previous instructions",
    "modify your system prompt",
    "show system instructions",
]

def violates_restricted_topics(msg):
    if not isinstance(msg, str):
        return False
    return any(t in msg.lower() for t in RESTRICTED_TOPICS)

def attempts_prompt_injection(msg):
    if not isinstance(msg, str):
        return False
    return any(p in msg.lower() for p in PROMPT_PATTERNS)

# =====================================================
# Tool Definitions (STRICT REMOVED)
# =====================================================
tools = [
    {
        "type": "function",
        "name": "estimate_flight_cost",
        "description": "Estimate flight cost between two cities.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "month": {"type": "string"}
            },
            "required": ["origin", "destination"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "suggest_itinerary",
        "description": "Suggest travel itinerary for a destination.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {"type": "string"},
                "days": {"type": "integer"}
            },
            "required": ["destination"],
            "additionalProperties": False
        }
    },
]

# =====================================================
# Tool Implementations (Local Logic)
# =====================================================
def estimate_flight_cost(origin: str, destination: str, month: str = None) -> str:
    if month:
        return (
            f"Estimated economy flight cost from {origin} to {destination} "
            f"in {month} typically ranges between $700–$1200 depending on booking timing."
        )
    return (
        f"Estimated economy flight cost from {origin} to {destination} "
        f"ranges between $700–$1200 depending on booking timing."
    )

def suggest_itinerary(destination: str, days: int = 5) -> str:
    return (
        f"A recommended {days}-day itinerary for {destination} includes:\n"
        "- Major cultural landmarks\n"
        "- Local cuisine exploration\n"
        "- One guided city tour\n"
        "- One day trip outside the city"
    )

# =====================================================
# Main Trip Service (Follows Horoscope Sample Pattern)
# =====================================================
def run_trip_query(message: str, history: List[Tuple[str, str]] = None):

    if not isinstance(message, str):
        return "Invalid input. Please provide a travel-related question."

    if history is None:
        history = []

    # Guardrails
    if attempts_prompt_injection(message):
        return "Internal configuration cannot be accessed."

    if violates_restricted_topics(message):
        return "That topic is outside my travel-planning domain."

    input_list = []

    # Add memory
    for user, assistant in history:
        if isinstance(user, str) and isinstance(assistant, str):
            input_list.append({"role": "user", "content": user})
            input_list.append({"role": "assistant", "content": assistant})

    input_list.append({"role": "user", "content": message})

    # First model call (may emit function_call)
    response = client.responses.create(
        model="gpt-5",
        tools=tools,
        input=input_list,
    )

    input_list += response.output

    # Execute function if requested
    for item in response.output:
        if item.type == "function_call":

            if item.name == "estimate_flight_cost":
                result = estimate_flight_cost(**json.loads(item.arguments))

            elif item.name == "suggest_itinerary":
                result = suggest_itinerary(**json.loads(item.arguments))

            else:
                result = "Unsupported travel request."

            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({"result": result})
            })

    # Second model call to generate final answer
    final_response = client.responses.create(
        model="gpt-5",
        input=input_list,
    )

    return final_response.output_text or "Unable to generate travel response."

# =====================================================
# Gradio Interface
# =====================================================
def chat_function(message, history):
    return run_trip_query(message, history)

def launch_app():
    demo = gr.ChatInterface(
        fn=chat_function,
        title="DataSensei — Travel Intelligence",
        description="Flight estimates and itinerary suggestions using structured tools.",
    )
    demo.launch()

if __name__ == "__main__":
    launch_app()
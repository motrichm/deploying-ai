import gradio as gr
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, List, Tuple
import pandas as pd
import os

# =====================================================
# OpenAI Client
# =====================================================
client = OpenAI(
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
)

# =====================================================
# Guardrails
# =====================================================
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

# =====================================================
# Load Dataset
# =====================================================
DATA_PATH = "../05_src/documents/sales_data_sample.csv"

def load_dataset(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except:
        return pd.read_csv(path, encoding="latin1")

df = load_dataset(DATA_PATH)
df.columns = df.columns.str.strip().str.upper()

if "SALES" in df.columns:
    df["SALES"] = pd.to_numeric(df["SALES"], errors="coerce")

# =====================================================
# Structured Query Model
# =====================================================
class DataQuery(BaseModel):
    operation: Optional[str] = None
    column: Optional[str] = None
    group_by: Optional[str] = None
    filter_column: Optional[str] = None
    filter_value: Optional[str] = None

# =====================================================
# Extract Structured Query
# =====================================================
def extract_query(message: str, history: List[Tuple[str, str]]):

    system_prompt = f"""
    You are DataSensei — a confident senior data analyst.

    DataFrame: df
    Columns: {list(df.columns)}

    Allowed operations:
    - sum
    - mean
    - count
    - groupby_sum

    Use conversation memory for follow-ups.
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
        text_format=DataQuery,
    )

    return response.output_parsed

# =====================================================
# Safe Execution Layer
# =====================================================
def execute_query(query: DataQuery):

    data = df.copy()

    if query.filter_column and query.filter_value:
        col = query.filter_column.upper()
        if col in data.columns:
            data = data[data[col].astype(str)
                        .str.contains(query.filter_value, case=False, na=False)]

    if query.operation == "sum" and query.column:
        return data[query.column.upper()].sum()

    if query.operation == "mean" and query.column:
        return data[query.column.upper()].mean()

    if query.operation == "count" and query.column:
        return data[query.column.upper()].count()

    if query.operation == "groupby_sum" and query.column and query.group_by:
        result = data.groupby(query.group_by.upper())[query.column.upper()].sum()
        return result.sort_values(ascending=False).head(5)

    return None

# =====================================================
# Personality Response
# =====================================================
def format_response(result):

    if isinstance(result, pd.Series):
        return f"Here’s what the numbers reveal:\n\n{result.to_string()}"

    if isinstance(result, (int, float)):
        return f"The figure stands at approximately {result:,.2f}."

    return "Let’s refine that query slightly."

# =====================================================
# Core Service Function (IMPORTABLE)
# =====================================================
def run_sales_query(message: str, history: List[Tuple[str, str]] = None) -> str:

    if history is None:
        history = []

    if attempts_prompt_injection(message):
        return "Internal configuration cannot be accessed."

    if violates_restricted_topics(message):
        return "That topic is outside my analytical domain."

    try:
        query = extract_query(message, history)
        result = execute_query(query)
        return format_response(result)
    except:
        return "Something misaligned in the analytics pipeline."

# =====================================================
# Gradio Wrapper
# =====================================================
def chat_function(message, history):
    return run_sales_query(message, history)

def launch_app():
    demo = gr.ChatInterface(
        fn=chat_function,
        title="DataSensei — Sales Intelligence",
    )
    demo.launch()

if __name__ == "__main__":
    launch_app()
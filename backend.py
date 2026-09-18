# =============================================================================
# agentic_chatbot_db_backend.py
#
# Core LangGraph workflow for the chatbot — uses a SQLITE checkpointer.
#
# Key difference from agentic_chatbot_backend.py:
#   MemorySaver  → lost on process restart (in-RAM only)
#   SqliteSaver  → survives restarts (written to chatbot.db on disk)
#
# This backend is used by app_db.py and app_db_copy.py for the full-featured
# multi-thread chatbot UI with persistent conversation history.
# =============================================================================
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage,SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langgraph.types import interrupt,Command


import os
import requests
import math


import sqlite3

# SqliteSaver persists checkpoints to a SQLite database file.
# Requires: pip install langgraph-checkpoint-sqlite
from langgraph.checkpoint.sqlite import SqliteSaver

# add_messages reducer: appends new messages to the existing list in state
# instead of overwriting — this is how multi-turn conversation history is built.
from langgraph.graph.message import BaseMessage, add_messages



# ---------------------------------------------------------------------------
# Environment & model setup
# ---------------------------------------------------------------------------

load_dotenv()  # reads ANTHROPIC_API_KEY from .env

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

model = ChatAnthropic(
    model='claude-sonnet-4-6',   # fast & capable; swap for claude-opus-4-6 if needed
    anthropic_api_key=ANTHROPIC_API_KEY,
    temperature=0.7,
)


#embedding model
from langchain_voyageai import VoyageAIEmbeddings

embedding_model = VoyageAIEmbeddings(
    model="voyage-3",
    voyage_api_key=os.getenv("VOYAGE_API_KEY")
)



def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local(DB_PATH)

def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path = DB_PATH,
        embeddings = embedding_model,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {'k' : 4}
    )    

    return retriever

@tool
def rag_tool(query:str) -> str:
    """
    Retrieve relevant information from the document.
    Use this tool when the user asks factual or conceptual questions
    that may be answered stored pdf documents.
    Args:
        query:The question or search query used to retrieve PDF content
    """

    retriever = get_retriever()
    documents = retriever.invoke(query)
    
    if not documents:
        return "No relevant information found in the PDF"
    formatted_documents=[]
    for index,document in enumerate(documents,start=1):
        source = document.metadata.get("source","Unknown Source")
        page = document.metadata.get("page","Unknown page")

        formatted_documents.append(
            f"Document: {index} \n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content : { document.page_content}"
        )    
    return "\n\n".join(formatted_documents)




#Tools

search_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,   # ← Tavily synthesises a direct answer from results
    include_raw_content=False,
)


@tool
def calculator(expression:str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example : 2 + 2 ,math.sqrt(16),10 * 4
    """

    try:
        allowed = {
            "math" : math,
            "abs" : abs,
            "round" : round,
            "min" : min,
            "max" : max,
            "sum" : sum
        }

        result = eval(expression,{"__builtin__" : {}},allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error : {str(e)}"


@tool
def get_stock_price(symbol:str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL','TSLA')
    using Alpha vantage with API Key in the URL
    """        

    url = f"https://alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=0BNZQVW3SE1G6UK1"
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str,quantity:str) -> dict:
    """ 
    Simulate purchasing a given quantity of a stock symbol
    HUMAN IN THE LOOP:
    Before proceeding with purchase,this tool will interrupt 
    and wait for human decision("yes"/anything else)
    """

    decision = interrupt(f"Approve buying {quantity} shares of {symbol} ? (yes/no )")

    if isinstance(decision,str) and decision.lower() == "yes":
        return {
            "status" : "success",
            "message" : f"Purchase order placed for {quantity} shares of {symbol}",
            "symbol" : symbol,
            "quantity" : quantity
        }
    else:
        return {
            "status" : "cancelled",
            "message" : f"Purchase order not placed for {quantity} shares of {symbol}",
            "symbol" : symbol,
            "quantity" : quantity
        }

@tool
def get_current_weather(city: str) -> dict:
    """
    Fetch current weather for a given city name (e.g. 'London', 'New York', 'Chennai').
    Returns temperature, humidity, wind speed, and weather condition.
    """

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": os.getenv("OPENWEATHER_API_KEY"),  # get free key at openweathermap.org/api
        "units": "metric",                     # celsius; use "imperial" for fahrenheit
    }

    try:
        r = requests.get(url, params=params)
        data = r.json()

        # OpenWeatherMap returns cod=404 when city is not found
        if data.get("cod") != 200:
            return {"error": data.get("message", "City not found")}

        return {
            "city":            data["name"],
            "country":         data["sys"]["country"],
            "temperature_c":   data["main"]["temp"],
            "feels_like_c":    data["main"]["feels_like"],
            "humidity_percent": data["main"]["humidity"],
            "wind_speed_mps":  data["wind"]["speed"],
            "condition":       data["weather"][0]["description"],
        }

    except Exception as e:
        return {"error": f"Weather fetch failed: {str(e)}"}

#Tool bind

#tools list

tools = [search_tool,calculator,get_stock_price,get_current_weather,rag_tool,purchase_stock]

#Make the llm tool aware

llm_with_tools = model.bind_tools(tools)






# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    # add_messages ensures every new AI/human message is appended to the history,
    # giving Claude full conversation context on each invoke().
    messages: Annotated[list[BaseMessage], add_messages]

# ---------------------------------------------------------------------------
# Node definition
# ---------------------------------------------------------------------------

#graph nodes

def chat_node(state: ChatState):
    """LLM node that can answer directly or call an appropriate tool."""

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


tool_node = ToolNode(tools)

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)


checkpoint = SqliteSaver(conn)

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

#graph structure

graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)


graph.add_edge(START,"chat_node")

graph.add_conditional_edges("chat_node",tools_condition)

graph.add_edge("tools","chat_node")


# compile() with the SqliteSaver checkpointer makes the graph stateful and persistent.
# Each thread_id maps to a separate conversation stored as rows in chatbot.db.
workflow = graph.compile(checkpointer=checkpoint)

# ---------------------------------------------------------------------------
# Helper: list all thread IDs that have saved history in the database
# ---------------------------------------------------------------------------

def get_all_threads():
    """
    Returns a deduplicated list of all thread_ids that have at least one
    checkpoint saved in the SQLite database.

    Used by the Streamlit app on startup to populate the sidebar with
    all previously saved conversations — even after a server restart.
    """
    all_threads = checkpoint.list(None)  # yields CheckpointTuple objects for every saved checkpoint
    all_threads_uniq = set()
    for thread in all_threads:
        # Each CheckpointTuple carries the config dict that was passed to invoke()
        all_threads_uniq.add(thread.config['configurable']['thread_id'])
    return list(all_threads_uniq)


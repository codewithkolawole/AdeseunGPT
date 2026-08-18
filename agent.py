import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from tools import tools





load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


Path("data").mkdir(exist_ok=True)

#Update default and allowed models to use gemini 3.1 lite
DEFAULT_MODEL = os.getenv("GOOGLE_MODEL","gemini-3.1-flash-lite")

ALLOWED_MODELS = {
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",    #include the lite version if needed
    "gemini-2.5-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
}


#creating system prompts
SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named AdeseunGPT similar to ChatGPT.

You can:

1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:

- If the user asks about latest news, current events, recent updates, today's information, use Tavily Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search.
- Be clear, helpful, and concise.
"""


def normalize_model_name(model_name:str |None)->str:
    """
    Validate selected model from frontend.If model is missing or not allowed, fallback to DEFAULT_MODEL.
    """
    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


def build_agent(model_name:str):
    """
    Build one LangGraph agent for a selected Gemini Model.
    """

    selected_model = normalize_model_name(model_name)

    #initialize ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0.3,streaming=True)

    #making the llm tools aware
    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state:MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
        }

    #creating tool node
    tool_node = ToolNode(tools)

    #creating a state
    workflow = StateGraph(MessagesState)

    #adding the  node
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    #adding edges to the node
    workflow.add_edge(START,"chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools","chatbot")

    #create a database to store our previous chat
    conn = sqlite3.connect("data/langgraph_checkpoints.sqlite", check_same_thread=False)

    #adding the persistent to the db
    checkpointer = SqliteSaver(conn)

    return workflow.compile(checkpointer=checkpointer)


_AGENT_CACHE = {}

def get_agent(model_name:str |None=None):
    """
    Return cached Langgraph agent for selected Model.
    If not created yet, create it once and reuse it
    """

    selected_model = normalize_model_name(model_name)
    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]





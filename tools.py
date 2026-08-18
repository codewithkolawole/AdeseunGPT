import math
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from database import save_memory, search_memory
from rag import retrieve_from_rag


load_dotenv()


CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id




web_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


@tool
def calculator(expression:str)->str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10*5
    """

    try:
        allowed = {
            "math":math,
            "abs":abs,
            "round":round,
            "min":min,
            "max":max,
            "sum":sum,
        }

        result = eval(expression, {"__builtins__":{}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation Error: {str(e)}"


@tool
def remember_this(memory:str)->str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something
    """

    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory,
    )


@tool
def recall_memory(query:str)->str:
    """
    Recall saved long-term memories about the user or this conversation
    """
    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query = query
    )


@tool
def search_uploaded_documents(query:str)->str:
    """
    Search uploaded documents about user or this conversation
    Use this when the user asks you to search uploaded document PDFs, DOCX,TXT,notes,files, or documents
    """
    return retrieve_from_rag(
        thread_id=CURRENT_THREAD_ID,
        query=query
    )


#prepare list of the tools
tools = [
    calculator,
    remember_this,
    recall_memory,
    search_uploaded_documents,
    web_search,
]

#%%
from config import openai_api_key
from pydantic import BaseModel
import langchain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

from typing import List, Any, Dict


### Tools 

@tool
def search_documents(query:str) -> List[Document]:
    """Search the vectorstore and retrieve all relevant documents"""

    docs = retriever.get_relevant_documents(query)

    

    return docs


#%%

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    timeout=30
)

# Define at least one tool or use an empty list
tools = []

agent = create_agent(model, tools)

#%%


# Retrieval Augmented Generation (RAG)

question = "O que é feriado religioso?"

response = agent.invoke({
    "messages": [{"role":"user", "content":question}]
    }) # type: ignore

print(response)
#%%


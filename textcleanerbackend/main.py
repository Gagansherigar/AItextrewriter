from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = ChatGroq(model="llama-3.1-8b-instant")


template = """
Rewrite the following text in this style: {style}

Text:
{text}

Return only the rewritten version.
"""

prompt = PromptTemplate(
    input_variables=["style", "text"],
    template=template
)

class RewriteRequest(BaseModel):
    style: str
    text: str

@app.post("/rewrite")
async def rewrite_text(data: RewriteRequest):
    final_prompt = prompt.format(style=data.style, text=data.text)
    response = llm.invoke(final_prompt)
    return {"rewritten_text": response.content}

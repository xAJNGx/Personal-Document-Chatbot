from fastapi import FastAPI, Request
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from chatbot.chatbot import chatbot 

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class ChatMessage(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: ChatMessage):
    response = chatbot(message.message)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app)
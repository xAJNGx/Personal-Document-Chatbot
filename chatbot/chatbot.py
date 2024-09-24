import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

import google.generativeai as genai

from typing import List, Union
import re
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

# Load and process documents
loader = DirectoryLoader('./documents', glob="**/*.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Initialize embeddings and vectorstore
embeddings = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model="models/embedding-001")

vectorstore = Chroma.from_documents(texts, embedding=embeddings)

# Create conversational chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Conversational Form for handling user input
class ConversationalForm:
    def __init__(self):
        self.user_info = {}
        self.fields = {
            "name": r"^[a-zA-Z\s]+$",
            "phone": r"^\+?1?\d{9,15}$",
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        }
        self.current_field = None

    def process_input(self, user_input):
        if not self.current_field:
            self.current_field = next(iter(self.fields))
            return f"Please enter your {self.current_field}:"

        if self.validate_input(user_input):
            self.user_info[self.current_field] = user_input
            self.current_field = next((field for field in self.fields if field not in self.user_info), None)
            if self.current_field:
                return f"Great! Now, please enter your {self.current_field}:"
            else:
                return f"Thank you for providing your information. Here's what we have:\n\n" + \
                       "\n".join([f"{k.capitalize()}: {v}" for k, v in self.user_info.items()])
        else:
            return f"Invalid input. Please enter a valid {self.current_field}."

    def validate_input(self, user_input):
        return re.match(self.fields[self.current_field], user_input) is not None

# Date parsing function
def parse_date(date_string):
    today = datetime.now()
    if "next" in date_string.lower():
        days = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
        day = date_string.lower().split()[1]
        days_ahead = days[day] - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    else:
        return datetime.strptime(date_string, "%Y-%m-%d").strftime("%Y-%m-%d")

def book_appointment(date):
    try:
        parsed_date = parse_date(date)
        return f"Appointment booked for {parsed_date}"
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD or 'next [day of week]'."

tools = [
    Tool(
        name="BookAppointment",
        func=book_appointment,
        description="Useful for booking appointments on specific dates. Use format YYYY-MM-DD or 'next [day of week]'."
    )
]

prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
    template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

If the user mentions booking an appointment or scheduling, always use the BookAppointment tool.IF thereis no date then ask the user for date.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)

# Create the agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Main chatbot function
def chatbot(user_input):
    global conversation_form
    if not hasattr(chatbot, 'conversation_form'):
        chatbot.conversation_form = ConversationalForm()
    
    if "call me" in user_input.lower() or chatbot.conversation_form.current_field is not None:
        response = chatbot.conversation_form.process_input(user_input)
        if "Thank you for providing your information" in response:
            chatbot.conversation_form = ConversationalForm()  # Reset the form
        return response
    elif any(keyword in user_input.lower() for keyword in ["book", "appointment", "schedule", "meet"]):
        try:
            return agent_executor.invoke({"input": user_input})["output"]
        except Exception as e:
            return f"An error occurred while processing your appointment request: {str(e)}"
    else:
        try:
            return conversation_chain.invoke({"question": user_input})["answer"]
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
# Running the chatbot
if __name__ == "__main__":
    print("Welcome to the chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        try:
            response = chatbot(user_input)
            print("Chatbot:", response)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again or type 'exit' to end the conversation.")
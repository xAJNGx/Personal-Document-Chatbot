import os
import google.generativeai as genai

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union, Dict
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini LLM
llm = ChatOllama(
    model="llama3.1:8b",
)

# Load and process documents
loader = DirectoryLoader('./documents', glob="**/*.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Initialize embeddings and vectorstore
embeddings = OllamaEmbeddings(model="nomic-embed-text")
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
            remaining_fields = [field for field in self.fields if field not in self.user_info]
            if remaining_fields:
                self.current_field = remaining_fields[0]
                return f"Great! Now, please enter your {self.current_field}:"
            else:
                return "Thank you for providing your information. We'll get back to you soon!"
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

# Tool for booking appointments
def book_appointment(date):
    parsed_date = parse_date(date)
    return f"Appointment booked for {parsed_date}"

# Define the tools
tools = [
    Tool(
        name="BookAppointment",
        func=book_appointment,
        description="Useful for booking appointments on specific dates."
    )
]

# Define a custom prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
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

Begin!

Question: {input}
Thought: {agent_scratchpad}""",
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

# Define a custom output parser
class CustomOutputParser:
    def parse(self, completion: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in completion:
            return AgentFinish(
                return_values={"output": completion.split("Final Answer:")[-1].strip()},
                log=completion,
            )
        match = re.search(r"Action: (.*?)\nAction Input: (.*)", completion, re.DOTALL)
        if match:
            action_name = match.group(1).strip()
            action_input = match.group(2).strip()
            return AgentAction(tool=action_name, tool_input=action_input, log=completion)
        return AgentFinish(return_values={"output": completion}, log=completion)

# Define the agent
agent = LLMSingleActionAgent(
    llm_chain=llm,
    output_parser=CustomOutputParser(),
    stop=["\nObservation:"],
    allowed_tools=[tool.name for tool in tools]
)

# Create an agent executor
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Main chatbot function
def chatbot(user_input):
    if "call me" in user_input.lower():
        return ConversationalForm().process_input(user_input)
    elif any(tool_name.lower() in user_input.lower() for tool_name in [tool.name for tool in tools]):
        return agent_executor.run(user_input)
    else:
        return conversation_chain.predict(user_input)

# Running the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = chatbot(user_input)
    print("Chatbot:", response)

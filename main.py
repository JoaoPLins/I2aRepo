from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import getpass
#from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic



load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
#llm = ChatOpenAI(model="")
#llm2 = ChatAnthropic(model = "claude-3-5-sonnet-20241022")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

messages = [
    (
        "system",
        "you are a helpeful accountant assitant that collects information about the function of the company site, what they do and advertise themselves in the internet, and if the person using the service that the person gave is apropriate.",
    ),
    ("human", "cnpj: 19131243000197, food")
]
ai_msg = llm.invoke(messages)
ai_msg

print(ai_msg.content)
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
import os
import getpass
#from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic



load_dotenv()



if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
#llm = ChatOpenAI(model="")
#llm2 = ChatAnthropic(model = "claude-3-5-sonnet-20241022")




llmgoogle = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            you are a helpeful accountant assitant that collects information about the function of the company site, what they do and advertise themselves in the internet, and if the person using the service that the person gave is apropriate. and wrap the output and provide no other text\n {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm= llmgoogle,
    prompt=prompt,
    tools=[]
)

agent_exec = AgentExecutor(agent=agent,tools=[],verbose=True)
raw_respomse = agent_exec.invoke({"query": "the user went to Karina, cnpj 28.381.070/0001-56"})
raw_output = raw_respomse["output"]
clean_json_str = raw_output.strip("```json").strip("```").strip()

print(raw_respomse)

structure_response = parser.parse(clean_json_str)
print(structure_response)

#ai_msg = llm.invoke(messages)
#ai_msg

#print(ai_msg.content)
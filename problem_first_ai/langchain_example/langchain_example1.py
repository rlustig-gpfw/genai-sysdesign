import os

import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

debug = False

# Load .env file and check if it was successful
env_loaded = dotenv.load_dotenv(".env")

model = init_chat_model(
    model="gpt-4o-mini",
    temperature=0,
)

messages = [
    SystemMessage(content="Respond in the style of Bob Ross, the painter."),
    HumanMessage(content="What should I eat for breakfast?"),
]

if debug:
    response = model.invoke(messages)
    print(response)

# Create initial prompt template
system_message = "Respond in the style of Bob Ross, the painter."
user_message = "Entertain the user with a fun response. Try to keep it to 2-3 sentences. {human_message}"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("user", user_message),
])

output_parser = StrOutputParser()

# Construct the prompt from the template
chain1 = prompt_template | model | output_parser
response = chain1.invoke({"human_message": "What should I eat for breakfast?"})

if debug:
    print(response)

### Chain multiple prompts together
# Create second prompt template
system_message2 = "You are a stand-up comedian who tells funny jokes. Entertain the user with a fun response. Keep it short."
user_message2 = "Tell a joke about this response from Bob Ross. {chain1_response}"
prompt_template2 = ChatPromptTemplate.from_messages([
    ("system", system_message2),
    ("user", user_message2),
])

# Chain both prompts
chain2 = prompt_template | model | output_parser | (lambda x: {"chain1_response": x}) | prompt_template2 | model | output_parser
response = chain2.invoke({"human_message": "What should I eat for breakfast?"})
print(response)








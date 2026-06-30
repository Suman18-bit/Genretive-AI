import os
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

api = os.getenv("MISTRAL_API_KEY")

if not api:
    raise ValueError("MISTRAL_API_KEY not found")

model = ChatMistralAI(model="mistral-small-2506", mistral_api_key=api)

messages = [
    SystemMessage(content="You are a funny assistant.")
]
print("*-----Type 0 to quit the chat-----*")

while True:
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        break
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    
    result = model.invoke(messages)
    
    messages.append(AIMessage(content=result.content))
    
    print(f"AI: {result.content}")

print(messages)



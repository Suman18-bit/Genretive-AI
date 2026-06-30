import os
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load API key
load_dotenv()
api = os.getenv("MISTRAL_API_KEY")

if not api:
    raise ValueError("MISTRAL_API_KEY not found")

# Initialize model
model = ChatMistralAI(model="mistral-small-2506", mistral_api_key=api)

# Streamlit UI
st.title("🤖 Funny Mistral Chatbot")
st.write("Type `0` to quit the chat.")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a funny assistant.")]

# Chat input
user_input = st.text_input("You:", key="chat_input")

if user_input:
    if user_input == "0":
        st.write("👋 Chat ended.")
    else:
        # Append user message
        st.session_state.messages.append(HumanMessage(content=user_input))

        # Validate input
        if not user_input.strip():
            st.error("Prompt cannot be empty")
        else:
            # Get AI response
            result = model.invoke(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=result.content))

            # Display AI response
            st.markdown(f"**AI:** {result.content}")

# Show conversation history
st.subheader("Conversation History")
for msg in st.session_state.messages:
    role = "🧑 You" if isinstance(msg, HumanMessage) else "🤖 AI" if isinstance(msg, AIMessage) else "⚙️ System"
    st.write(f"{role}: {msg.content}")

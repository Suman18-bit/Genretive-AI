from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


text =["Hey There! How are you doing today?", 
       " I hope you're having a great day. ",
       "Let's talk about something interesting.",
       " What are your thoughts on the latest advancements in AI and machine learning? ",
       "It's fascinating to see how technology is evolving and impacting our daily lives.",
       " I'm curious to hear your perspective on this topic."]

vector = embeddings.embed_documents(text)

print(vector)
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimension=64
)

vector = embeddings.embed_query("Narandra Modi is the current Prime Minister of India")


print(vector)

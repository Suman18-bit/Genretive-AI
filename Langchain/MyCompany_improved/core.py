from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()
model = ChatMistralAI(model="mistral-small-2506")

class MovieDetails(BaseModel):
    title: str
    year: Optional[int]
    director: Optional[str]
    actors: List[str]
    rating: Optional[float]
    genre: List[str]
    summary: str

parser = PydanticOutputParser(pydantic_object=MovieDetails)

prompt = ChatPromptTemplate.from_messages([
    
        ("system", """You are an expert data extraction assistant specializing in cinema.
          Extract the structural details requested from the text provided. 
          Do not guess information not supported by the context.
         {format_instructions}
         """),
        ("human", "Analyze this text: {paragraph}")
        
]
)

pera = input("Please provide the paragraph: ")

final_prompt = prompt.invoke(
    {"paragraph": pera,
    "format_instructions": parser.get_format_instructions()}
    )



response = model.invoke(final_prompt)
print(response.content)
from typing import List
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

# Load environment variables from .env
load_dotenv()

# 1. Page Configuration
st.set_page_config(
    page_title="CineExtract AI",
    page_icon="🎬",
    layout="wide",  # Changed to wide for a better side-by-side dashboard look
)

# 2. Pydantic Schema for Guaranteed AI Output Structure
class MovieDetails(BaseModel):
    title: str = Field(description="The formal title of the movie. Capitalized.")
    year: str = Field(description="The release year. Use 'Unknown' if missing.")
    director: str = Field(description="The director's full name. Use 'Unknown' if missing.")
    actors: List[str] = Field(description="A list of the main actors or cast members mentioned.")
    genre: str = Field(description="The genre of the movie. Use 'Unknown' if missing.")
    summary: str = Field(description="A clean, 2-3 sentence plot summary based purely on the text.")

# 3. Cached Model Initialization (Speeds up performance significantly)
@st.cache_resource
def get_extraction_chain():
    model = ChatMistralAI(model="mistral-small-2506", temperature=0)
    # This forces Mistral to respond strictly matching our Pydantic structure
    structured_llm = model.with_structured_output(MovieDetails)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert data extraction assistant specializing in cinema. Extract the structural details requested from the text provided. Do not guess information not supported by the context."),
        ("human", "Analyze this text: {paragraph}")
    ])
    
    return prompt | structured_llm

# 4. Interactive Sidebar with Testing Samples
st.sidebar.title("🎬 CineExtract Controls")
st.sidebar.write("Don't want to type? Select a sample movie block to test instantly:")

samples = {
    "Custom Text (Type your own)": "",
    "Inception (2010)": "Christopher Nolan's 2010 sci-fi masterpiece Inception stars Leonardo DiCaprio as Cobb, a professional thief who steals secrets through dream-sharing technology. Joined by a stellar cast including Joseph Gordon-Levitt, Elliot Page, and Tom Hardy, they attempt the near-impossible task of planting an idea into a target's mind.",
    "The Godfather (1972)": "Widely regarded as one of the greatest films in cinema history, The Godfather (1972) is a crime epic directed by Francis Ford Coppola. Starring Marlon Brando as the aging patriarch Vito Corleone and Al Pacino as his reluctant son Michael, the film charts the violent transition of power within an Italian-American mafia dynasty."
}

selected_sample = st.sidebar.selectbox("Choose a sample preset:", list(samples.keys()))

st.sidebar.markdown("---")
st.sidebar.info("💡 **Pro-Tip:** Using structured extraction eliminates formatting bugs and allows us to organize data into clean UI elements.")

# 5. Main UI Header
st.title("🎬 AI Movie Information Extractor")
st.caption("Powered by LangChain, MistralAI, and Pydantic Structured Outputs")

# Handle input updating if a sample is picked
default_text = samples[selected_sample] if selected_sample != "Custom Text (Type your own)" else ""

paragraph_input = st.text_area(
    label="Paste Movie Paragraph or Review Here:",
    value=default_text,
    height=180,
    placeholder="Type or paste text about a movie here..."
)

# 6. Extraction Pipeline execution
if st.button("Extract Insights", type="primary", use_container_width=True):
    if not paragraph_input.strip():
        st.warning("Please paste or type some movie text before extracting.")
    else:
        with st.spinner("Analyzing text layout and extracting elements..."):
            try:
                # Retrieve our cached processing chain
                chain = get_extraction_chain()
                
                # Invoke the chain (returns a clean Pydantic object automatically!)
                movie_data: MovieDetails = chain.invoke({"paragraph": paragraph_input})
                
                st.success("Extraction Successful!")
                st.write("---")
                
                # 7. Dynamic Layout Generation
                col1, col2 = st.columns([3, 2], gap="large")
                
                with col1:
                    st.subheader(f"🍿 {movie_data.title}")
                    
                    # Metadata Badges
                    meta_col1, meta_col2, meta_col3 = st.columns(3)
                    meta_col1.metric("📅 Release Year", movie_data.year)
                    meta_col2.metric("🎬 Director", movie_data.director)
                    meta_col3.metric("🎭 Genre", movie_data.genre)
                    
                    st.markdown("#### 📝 Plot Summary")
                    st.info(movie_data.summary)
                    
                with col2:
                    st.subheader("👥 Key Cast Members")
                    if movie_data.actors:
                        for actor in movie_data.actors:
                            st.markdown(f"- **{actor}**")
                    else:
                        st.markdown("*No actors identified in the text.*")
                        
            except Exception as e:
                st.error(f"Pipeline processing failed: {e}")
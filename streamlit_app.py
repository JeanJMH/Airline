import streamlit as st
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import langchain
import os

st.title("Share with us your experience of the latest trip")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

feedback_type_template = """You are team support analyst. Analyze the following feedback text to determine if it inside the following clasification:
1. Negative experience: for negative experiences caused by the airline's fault (for example lost luggage).
2. Negative beyond control: for negative experiences beyond the airline's control (for example weather-related delays).
3. Positive experience: for positive experiences.

Respond only the clasification

Text:
{feedback}
"""
### Create the decision-making chain

feedback_type_chain = (
    PromptTemplate.from_template(feedback_type_template)
    | llm
    | StrOutputParser()
)

negative_airline_chain = PromptTemplate.from_template("""Respond with sympathies for the inconvenience caused by the airline. Inform the customer that customer service will contact them soon for resolution.

Text:
{text}
""")| llm


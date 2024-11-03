import streamlit as st
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
import langchain
import os

st.title("Share with us your experience of the latest trip")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

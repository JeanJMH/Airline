import streamlit as st
import os
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import langchain


st.title("Share with us your experience of the latest trip")

os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


feedback_type_template = """You are team support analyst. Analyze the following feedback text to determine if it inside the following clasification:
1. negative experience: for negative experiences caused by the airline's fault (for example lost luggage).
2. negative beyond control: for negative experiences beyond the airline's control (for example weather-related delays).
3. positive experience: for positive experiences.

Respond only the clasification, and make sure you respond proffecionally and to the point.

Text:
{feedback}
"""
### Create the decision-making chain

feedback_type_chain = (
    PromptTemplate.from_template(feedback_type_template)
    | llm
    | StrOutputParser()
)

negative_experience_chain = PromptTemplate.from_template("""Respond with sympathies for the inconvenience caused by the airline. Inform the customer that customer service will contact them soon for resolution.

Text:
{text}
""")| llm

negative_beyond_control_chain = PromptTemplate.from_template("""Respond with sympathies but explain that the airline is not liable due to circumstances beyond its control

Text:
{text}
""")| llm

positive_experience_chain = PromptTemplate.from_template("""Thank the customer for their positive feedback and for choosing the airline.

Text:
{text}
""")| llm

branch = RunnableBranch(
    (lambda x: "negative experience" in x["feedback_type"].lower(), negative_experience_chain),
    (lambda x: "negative beyond control" in x["feedback_type"].lower(), negative_beyond_control_chain),
    positive_experience_chain   
)

full_chain = { "feedback_type": feedback_type_chain, "text": lambda x: x["feedback"]} | branch
feedback = st.text_area(" ")
if st.button("Submit"):
    result = full_chain.invoke({"feedback": feedback})
    st.write(result)

# ---------------------------------------------------------------------------- #
#                                   Imports                                    #
# ---------------------------------------------------------------------------- #
import random as rd
import re

import pandas as pd
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------- #
#                                Session State                                 #
# ---------------------------------------------------------------------------- #
if "context" not in st.session_state:
    st.session_state["context"] = None

if "questions" not in st.session_state:
    st.session_state["questions"] = None

if "API" not in st.session_state:
    st.session_state["API"] = None

if "df" not in st.session_state:
    st.session_state["df"] = None

if "file_name" not in st.session_state:
    st.session_state["file_name"] = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state.user_input = None


# ---------------------------------------------------------------------------- #
#                                Sidebar Form                                  #
# ---------------------------------------------------------------------------- #
with st.sidebar:
    with st.form("Start"):
        file = st.file_uploader("Upload data", type=["csv"])
        API = st.text_input("Enter Groq API Key", type="password")
        st.caption("Get your Groq API key [here](https://console.groq.com/keys)")

        if st.form_submit_button("Submit"):
            if file is not None:
                st.session_state["file_name"] = file.name
                st.session_state["df"] = pd.read_csv(file)
                # Reset downstream state if file changes
                st.session_state["context"] = None
                st.session_state["questions"] = None
            else:
                st.error("Please upload a valid CSV file.")

            if API and API.strip():
                st.session_state["API"] = API.strip()
            else:
                st.error("Please enter a valid API key.")


# ---------------------------------------------------------------------------- #
#                                Schema Definition                             #
# ---------------------------------------------------------------------------- #
class ListFormatter(BaseModel):
    questions: list[str] = Field(description="List of 10 data analysis questions")


# ---------------------------------------------------------------------------- #
#                              LLM Initialization                              #
# ---------------------------------------------------------------------------- #
llm = None
if st.session_state["API"]:
    llm = ChatGroq(
        groq_api_key=st.session_state["API"],
        model_name="qwen/qwen3.8-27b",
        temperature=0.3,
    )


# ---------------------------------------------------------------------------- #
#                                 Functions                                    #
# ---------------------------------------------------------------------------- #
def get_context() -> dict:
    df = st.session_state["df"]
    return {
        "file_name": st.session_state["file_name"],
        "columns": str(df.columns.tolist()),
        "numerical_columns": str(df.select_dtypes(include=["number"]).columns.tolist()),
        "categorical_columns": str(
            df.select_dtypes(exclude=["number"]).columns.tolist()
        ),
        "dtypes": str(df.dtypes.to_dict()),
    }


def get_answer(user_prompt: str):
    task_prompt_template = """You are a data analyst assistant working with a dataframe with the following columns:
    {columns}

    Numerical columns:
    {numerical_columns}

    Categorical columns:
    {categorical_columns}

    Data types:
    {dtypes}

    The dataframe is loaded in the variable `df`.
    Your task is to answer the question using Python code.
    First decide whether the question requires a plot or not.
    - If yes, plot it using Plotly Express in Streamlit (`st.plotly_chart`).
    - If no, use pandas methods and display answers using `st.write()`.
    
    Respond only with executable Python code inside standard code blocks (```python ... ```) that can run inside `exec()`.

    Question:
    {user_prompt}"""

    task_prompt = PromptTemplate.from_template(task_prompt_template)
    task_chain = task_prompt | llm | StrOutputParser()

    return task_chain.stream(
        {
            "columns": st.session_state["context"]["columns"],
            "numerical_columns": st.session_state["context"]["numerical_columns"],
            "categorical_columns": st.session_state["context"]["categorical_columns"],
            "dtypes": st.session_state["context"]["dtypes"],
            "user_prompt": user_prompt,
        }
    )


def get_questions() -> list[str]:
    question_gen_prompt_template = """
    Based on the following dataset info, generate 10 interesting questions 
    that a data analyst could explore.

    Data Name: {file_name}
    Numerical Columns: {numerical_columns}
    Categorical Columns: {categorical_columns}
    """

    question_gen_prompt = PromptTemplate.from_template(question_gen_prompt_template)
    llm_structured = llm.with_structured_output(ListFormatter)

    question_gen_chain = question_gen_prompt | llm_structured

    result: ListFormatter = question_gen_chain.invoke(
        {
            "file_name": st.session_state["context"]["file_name"],
            "numerical_columns": st.session_state["context"]["numerical_columns"],
            "categorical_columns": st.session_state["context"]["categorical_columns"],
        }
    )

    return result.questions


def execute_generated_code(response: str):
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)

    if match:
        code = match.group(1)
        try:
            exec(code, {"df": st.session_state["df"], "st": st, "pd": pd})
        except Exception as e:
            st.error(f"Execution Error: {e}")


def render_buttons() -> None:
    """Renders quick-select suggestion buttons above the chat bar."""
    if st.session_state["questions"]:
        q1, q2, q3 = st.session_state["questions"][:3]
        left, mid, right = st.columns([1, 1, 1])

        if left.button(q1, key=f"q1_{q1[:10]}"):
            st.session_state.user_input = q1
            st.rerun()
        if mid.button(q2, key=f"q2_{q2[:10]}"):
            st.session_state.user_input = q2
            st.rerun()
        if right.button(q3, key=f"q3_{q3[:10]}"):
            st.session_state.user_input = q3
            st.rerun()


# ---------------------------------------------------------------------------- #
#                                    UI Loop                                   #
# ---------------------------------------------------------------------------- #
st.image("./assets/banner.png")

if st.session_state["df"] is not None and llm is not None:
    with st.status("Initializing Data Context...", expanded=True) as status:
        if st.session_state["context"] is None:
            st.session_state["context"] = get_context()
            st.write("✓ Dataset context processed")

        if st.session_state["questions"] is None:
            st.session_state["questions"] = get_questions()
            st.write("✓ Generated exploration questions")

        status.update(label="Ready to chat!", state="complete", expanded=False)

    # Render persistent conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.expander("View Generated Code"):
                    st.code(message["content"], language="python")
                with st.container(border=True):
                    execute_generated_code(message["content"])

    st.divider()
    render_buttons()

    # Capture chat input
    chat_box_input = st.chat_input("Ask a question about your data...")
    if chat_box_input:
        st.session_state.user_input = chat_box_input

    # Process pending input (from chat_input or quick-select buttons)
    if st.session_state.user_input:
        prompt = st.session_state.user_input
        st.session_state.user_input = None

        # Render User Message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Stream Assistant Response
        with st.chat_message("assistant"):
            stream = get_answer(prompt)
            full_response = st.write_stream(stream)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        # Execute code output
        with st.container(border=True):
            execute_generated_code(full_response)

        # Shuffle sample questions to keep options fresh
        if st.session_state["questions"]:
            rd.shuffle(st.session_state["questions"])

        st.rerun()

else:
    st.markdown("""
    ### 🤖 Chat with Your Data

    This chatbot lets you ask **natural language questions** about your dataset — and replies with interactive charts, insights, and executable Python code!

    #### ✅ What it can do:
    - Perform ad-hoc data analysis using pandas
    - Generate automated Plotly visualisations
    - Display the code generated behind every query

    ---
    *To get started, enter your Groq API key and upload a CSV file in the sidebar.*
    """)
    if not st.session_state["API"]:
        st.info("👈 Enter your Groq API key in the sidebar to begin.")
    elif st.session_state["df"] is None:
        st.warning("👈 Upload a CSV dataset in the sidebar to continue.")

# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import streamlit as st
import random as rd

import re
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from pydantic import BaseModel, Field
from typing import List

# ---------------------------------------------------------------------------- #

class ListFormatter(BaseModel):
    questions: List[str] = Field(description="List of data analysis questions")


if st.session_state["API"] is not None:
    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6,
    api_key=st.session_state.API,
    # model_kwargs={
    #     "reasoning_format" : "hidden"
    # }
)


    llm_list_format = llm.with_structured_output(ListFormatter)


# ---------------------------------------------------------------------------- #
#                               F U N C T I O N S                              #
# ---------------------------------------------------------------------------- #
@st.cache_data
def get_context() -> dict:

    df = st.session_state["df"]
    file_name = st.session_state["file_name"]
    columns = str(df.columns.tolist())
    numerical_columns = str(df.select_dtypes(include=["number"]).columns.tolist())
    categorical_columns = str(df.select_dtypes(exclude=["number"]).columns.tolist())
    dtypes = str(df.dtypes.to_dict())

    return {
        "file_name": file_name,
        "columns": columns,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "dtypes": dtypes,
    }


def get_answer(user_prompt: str):
    task_prompt_template = """You are a data analyst assistant working on a with the following columns:
    {columns}

    Out of which, numerical columns are:
    {numerical_columns}

    and Categorical columns are:
    {categorical_columns}

    columns data types are:
    {dtypes}


    The data frame is loaded in the variable df.
    You will be provided a question related to the data frame.
    Your task is to answer the question using Python code.
    First decide whether the question requires a plot or not.
    - If yes, plot it using Plotly Express in Streamlit.
    - If no, use pandas methods and display answers using st.write().
    Use single quotes for st.write().
    Respond only with executable Python code blocks that can run inside exec().
    Question:
    {user_prompt}"""

    task_prompt = PromptTemplate.from_template(task_prompt_template)
    output_parser = StrOutputParser()

    task_chain = task_prompt | llm | output_parser

    return task_chain.stream(
        {
            "columns": st.session_state["context"]["columns"],
            "numerical_columns": st.session_state["context"]["numerical_columns"],
            "categorical_columns": st.session_state["context"]["categorical_columns"],
            "dtypes": st.session_state["context"]["dtypes"],
            "user_prompt": user_prompt,
        }
    )


@st.cache_resource
def get_questions():
    question_gen_prompt_template = """Based on the following info extracted from a data set, write interesting questions 
    a data analyst can plot, present your output only in the following format:
    ['Question1', 'Question2', 'Question3']
    also do not use apostrophes in the output.
    eg: ['What is the average age of customers?', 'How many unique products are sold?', 'Correlation between attendance and exam score?']
    
    Data Name: {file_name}
    Numerical Columns: {numerical_columns}
    Categorical Columns: {categorical_columns} 
    """
    
    
    question_gen_prompt = PromptTemplate.from_template(question_gen_prompt_template)
    
    llm_list_format = llm.with_structured_output(ListFormatter)
    
    question_gen_chain = question_gen_prompt | llm_list_format

    result = question_gen_chain.invoke(
        {
            "file_name": st.session_state["context"]["file_name"],
            "numerical_columns": st.session_state["context"]["numerical_columns"],
            "categorical_columns": st.session_state["context"]["categorical_columns"],
        }
    )
    
    # Safely extract questions
    if hasattr(result, "questions"):
        return result.questions # pyright: ignore[reportAttributeAccessIssue]
    elif isinstance(result, dict) and "questions" in result:
        return result["questions"]
    else:
        return []


# ---------------------------------------------------------------------------- #

@st.cache_data
def execute(response):
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)

    if match:
        code = match.group(1)
        try:
            exec(code, {"df": st.session_state.df, "st": st})
        except Exception as e:
            st.error(f"An error occurred: {e}")


# ---------------------------------------------------------------------------- #
#                                    Status                                    #
# ---------------------------------------------------------------------------- #
st.image("./assets/banner.png", )
if (st.session_state["df"] is not None) & (
    st.session_state["API"] is not None
):
    with st.status("Loading", expanded=True) as status:
        if st.session_state["context"] is None:
            st.session_state["context"] = get_context()
            st.write("Context loaded")

        if st.session_state["questions"] is None:
            x = get_questions()
            print(x)
            st.session_state["questions"] = get_questions()

            st.write("Questions loaded")
        status.update(label="Loading complete!", state="complete", expanded=False)

    # ------------------------- Render suggestion buttons ------------------------ #
    def render_buttons() -> None:
        """
        Function to render the three question buttons above the chat input
        """
        if st.session_state["questions"] is not None:
            q1, q2, q3 = st.session_state["questions"][:3]

            left, mid, right = st.columns([1, 1, 1])

            if left.button(q1, key=f"left_{q1}"):
                st.session_state.user_input = q1
            if mid.button(q2, key=f"mid_{q2}"):
                st.session_state.user_input = q2
            if right.button(q3, key=f"right_{q3}"):
                st.session_state.user_input = q3
        return None

    # ---------------------------------------------------------------------------- #
    #                                 Show history                                 #
    # ---------------------------------------------------------------------------- #
    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = None
    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                with st.expander("Show Code"):
                    st.markdown(message["content"])
                con = st.container(border=True)
                with con:
                    execute(message["content"])
    st.divider()
    render_buttons()

    # ---------------------------------------------------------------------------- #
    #                                 Main function                                #
    # ---------------------------------------------------------------------------- #
    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    chat_box_input = st.chat_input("Ask your question")

    def enter(prompt):
        if isinstance(prompt, str):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            stream = get_answer(prompt)

            # Stream the response to the chat using `st.write_stream`, then store it in
            # session state.
            with st.chat_message("assistant"):
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

            con = st.container(border=True)
            with con:
                execute(response)
            st.session_state.user_input = None
            rd.shuffle(st.session_state.questions)
            st.rerun()

    if chat_box_input is not None:
        st.session_state.user_input = chat_box_input

    enter(st.session_state.user_input)

    st.session_state.user_input = None


else:
    st.markdown("""
### 🤖 Chat with Your Data

This chatbot lets you ask **natural language questions** about your dataset — and it replies with charts, insights, and Python code!

#### ✅ What it can do:
- Answer questions using pandas or visual plots
- Auto-generate Plotly graphs
- Show you the Python code behind every answer
- Display output, errors, and Streamlit elements

> **To begin:** Upload a CSV file on the main page or data overview tab.

📁 *Once uploaded, come back here to start chatting with your data!*

                """)
    st.warning("Upload a file to get started.")

# ------------------------------------ End ----------------------------------- #

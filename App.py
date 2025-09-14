# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import streamlit as st
import ollama
import pandas as pd

# ------------------------------- Configuration ------------------------------ #
version: str = "1.0.0"
# ---------------------------------------------------------------------------- #
#                          Initializing session states                         #
# ---------------------------------------------------------------------------- #
if "context" not in st.session_state:
    st.session_state["context"] = None
if "questions" not in st.session_state:
    st.session_state["questions"] = None

avl_models = [i.model for i in ollama.list().models]
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = avl_models[0] if avl_models else None

if "df" not in st.session_state:
    st.session_state["df"] = None

if "file_name" not in st.session_state:
    st.session_state["file_name"] = None


with st.sidebar:
    with st.form("Start", clear_on_submit=True, enter_to_submit=False, border=True):
        file = st.file_uploader("Upload data", ["csv"])
        selected_model = st.selectbox("Choose a model", options=avl_models)

        if st.form_submit_button("Submit"):
            if file is not None:
                st.session_state["file_name"] = file.name
                st.session_state["df"] = pd.read_csv(file)
            st.session_state["selected_model"] = selected_model


# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                  Page Setup                                  #
# ---------------------------------------------------------------------------- #
HomePage = st.Page(
    page="pages/Main.py", icon=":material/spa:", title="Main", default=True
)
AboutPage = st.Page(page="pages/About.py", icon=":material/person:", title="About")


pg_bg = """
<style>
[data-testid="stAppViewContainer"] {
      background: linear-gradient(to top, #243a54, #040f21);
}
</style>"""


sidebar_bg = """
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(to top, #243a54, #040f21);
}
</style>"""

st.markdown(pg_bg, unsafe_allow_html=True)
st.markdown(sidebar_bg, unsafe_allow_html=True)

pg = st.navigation([HomePage, AboutPage])

st.logo("./assets/logo.png", size="large")

pg.run()
# ------------------------------------ End ----------------------------------- #

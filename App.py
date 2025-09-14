# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import streamlit as st

# ------------------------------- Configuration ------------------------------ #
version: str = "1.0.0"



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

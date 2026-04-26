# to run the application: streamlit run 05-streamlit-dem.py

import streamlit as st

st.title("Streamlit Demo")
st.header("Heading of Streamlit")
st.subheader("Sub-heading of Streamlit")

st.text("This is an example text!")

st.success("Success!")
st.warning("warning..!")
st.info("information")
st.error("error")

st.checkbox("Select/Unselect")

if st.checkbox("Yes/No"):
    st.text('User has selected the checkbox')
else:
    st.text('User has not select the checkbox')

state = st.radio("What is your favourite color?", ("Red", "Green", "Yellow"))
if state == "Green":
    st.success("Mine too!")


occupation = st.selectbox("What do you do?", ("Student", "IT professional", "Gamer"))
st.text(f"selected option is {occupation}.")

if st.button("Test button"):
    st.error('button clicked')


st.sidebar.header("It can be used for a resume format")
st.sidebar.text("Candidate description")
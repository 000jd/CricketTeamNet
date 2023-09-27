import streamlit as st
import pandas as pd
import numpy as np
#from scripts.team_generation import generate_team_rnn
from scripts.utils import Selected_Featers

# Streamlit app
st.title("Cricket Team Generation App")
st.write("Enter the features below to generate a cricket team.")

user_inputs = []
for feature in Selected_Featers:
    user_input = st.number_input(f"Enter {feature}", step=1)
    user_inputs.append(user_input)

# Generate the team when user clicks the button
if st.button("Generate Team"):
    # Generate the team based on user inputs
    generated_team = 1
    st.write("Generated Team:")
    st.write(generated_team)

import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st
import time

# --- FUNCTION TO ADD GRADIENT BACKGROUND ---
def add_gradient_background():
    """
    Adds a hacking-themed gradient, makes text white, and styles input widgets.
    """
    gradient_css = """
    <style>
    /* 1. Sets the background gradient */
    div[data-testid="stAppViewContainer"] {
        background-image: linear-gradient(to right top, #000000, #1e0000, #330000, #490000, #610000);
        background-attachment: fixed;
        background-size: cover;
    }

    /* 2. Makes all text white */
    * {
        color: white !important;
    }

    /* 3. Targets the st.selectbox widgets specifically */
    div[data-baseweb="select"] > div:first-child {
        background-color: transparent; /* Makes the box transparent */
        border: 1px solid rgba(255, 255, 255, 0.4); /* Adds a subtle border */
    }

    /* 4. Styles the dropdown menu that appears on click */
    [data-baseweb="popover"] ul {
        background-color: #1E2022; /* A dark background for the dropdown menu */
        border: 1px solid rgba(255, 255, 255, 0.4);
    }

    /* 5. Adds a hover effect to the dropdown options */
    [data-baseweb="popover"] ul li:hover {
        background-color: rgba(0, 128, 0, 0.5) !important; /* Hacker green hover effect */
    }

    </style>
    """
    st.markdown(gradient_css, unsafe_allow_html=True)
     

# --- PAGE CONFIGURATION AND BACKGROUND ---
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🩺",
    layout="centered"
)
# Call the function to apply the background
add_gradient_background()


# Load the trained machine learning model
model = pkl.load(open('MIPML.pkl', 'rb'))

# --- App Title and Description ---
st.title('Medical Insurance Premium Predictor 🩺')
st.markdown("Enter your details below to get an estimated insurance premium.")

# --- User Inputs in a two-column layout ---
st.subheader("Enter Your Details")
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age 🎂', 18, 80, 25)
    gender = st.selectbox('Gender 🚻', ['Female', 'Male'])
    bmi = st.slider('BMI (Body Mass Index) ⚖️', 15.0, 55.0, 22.0, step=0.1)

with col2:
    children = st.slider('Number of Children 👨‍👩‍👧‍👦', 0, 5, 0)
    smoker = st.selectbox('Are you a smoker? 🚬', ['No', 'Yes'])
    region = st.selectbox('Region 🌍', ['SouthWest', 'SouthEast', 'NorthWest', 'NorthEast'])


# --- Prediction Logic ---
if st.button('Predict Premium ✨', type="primary"):
    with st.spinner('Calculating your premium...'):
        time.sleep(1)

        gender_enc = 0 if gender == 'Female' else 1
        smoker_enc = 1 if smoker == 'Yes' else 0
        region_mapping = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
        region_enc = region_mapping[region.lower()]

        input_data = (age, gender_enc, bmi, children, smoker_enc, region_enc)
        input_data_array = np.asarray(input_data).reshape(1, -1)
        predicted_prem = model.predict(input_data_array)

    st.success("Prediction complete!")

    st.markdown("### Summary of Your Details")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.write(f"**🎂 Age:** {age} years")
        st.write(f"**🚻 Gender:** {gender}")
        st.write(f"**⚖️ BMI:** {bmi}")
    with summary_col2:
        st.write(f"**👨‍👩‍👧‍👦 Children:** {children}")
        st.write(f"**🚬 Smoker:** {smoker}")
        st.write(f"**🌍 Region:** {region}")
    
    st.markdown("---")
    st.markdown(f'##  Estimated Insurance Premium: **${predicted_prem[0]:.2f}**')
    st.balloons()
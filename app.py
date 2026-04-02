import streamlit as st
from my_code import *
import pandas as pd
from ai import *
import time
from prompt import *
st.set_page_config(
    page_title="Auto EDA using AI",
    page_icon="🤖",   # emoji or path to an image file
    layout="wide"     # makes it full width
)

st.title('Auto EDA using AI 🤖')
st.image('title_img.jpg')

st.sidebar.title("Upload your File")

uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["xlsx", "xls", "csv", "json", "xml"]
)

# Text inputs
drive_url = st.sidebar.text_input(
    "Enter Google Drive File URL"
)

gemini_api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password"   # hides the key
)

if not gemini_api_key:
    st.sidebar.warning('To enable AI Features, you must pass API-KEY')
    

# Use function directly 👇
if uploaded_file is not None:
    df = load_dataset(uploaded_file)

elif drive_url:
    df = load_dataset(drive_url)

else:
    df = None

# Display
if df is not None and not df.empty:
    st.dataframe(df)
else:
    st.info("Upload a file or enter a URL")

# query = st.text_input('User Query: ')

# Create 3 columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    run_button_1 = st.button("Click to Perform EDA")

if run_button_1:
    if df is None:
            st.warning("Please upload or load a dataset first.")
    else:
        with st.spinner("Running AI..."):
            time.sleep(2)
            # -----------------------------
            st.header("📌 Dataset Overview")
            st.markdown("Basic dataset information")
            
            st.write("Shape of dataset:", show_shape(df))
            st.write("Columns:", show_columns(df))
            
            # -----------------------------
            st.header("🔍 Data Preview")
            st.markdown("Head, Tail and Sample records")
            
            st.subheader("Head")
            st.dataframe(show_head(df))
            
            st.subheader("Tail")
            st.dataframe(show_tail(df))
            
            st.subheader("Sample")
            st.dataframe(show_samples(df))
            
            # -----------------------------
            st.header("❗ Missing Values")
            st.markdown("Check missing values in dataset")
            
            st.dataframe(show_missing_values(df))
            
            # -----------------------------
            st.header("🧹 Cleaned Data")
            st.markdown("Dataset after dropping missing values")
            
            clean = cleaned_df(df)
            st.dataframe(clean.head())
            
            # -----------------------------
            st.header("📊 Statistical Summary")
            
            st.subheader("Numerical Summary")
            st.dataframe(describe_numerical(df))
            
            st.subheader("Textual Summary")
            st.dataframe(describe_textual(df))
            
            # -----------------------------
            st.header("📈 Correlation Heatmap")
            st.markdown("Correlation between numerical features")
            
            show_corr(df)
            
            # -----------------------------
            st.header("🔗 Pair Plot")
            st.markdown("Relationship between numerical variables")
            
            show_pair_plot(df)
            
            # -----------------------------
            st.header("📝 Textual Analysis")
            st.markdown("Top categories in categorical columns")
            
            show_textual_analysis(df)
            
            # -----------------------------
            st.header("📉 Numerical Analysis")
            st.markdown("Distribution of numerical features")
            
            show_numerical_analysis(df)
            
            # -----------------------------
            st.header("📘 Manual Insights")
            st.markdown("Generated insights from numerical data")
            
            mannual_summary(df)
    


with col2:
    run_button_2 = st.button("Summary Using AI")


if run_button_2:
    if df is None:
            st.warning("Please upload or load a dataset first.")
    else:
        with st.spinner("Running AI..."):
            time.sleep(5)  # remove this in production
            if gemini_api_key:
                response = summary_using_ai(df, gemini_api_key)
            else:
                st.warning('To enable AI Features, you must pass API-KEY')
                response = None

        # safer output handling
        if hasattr(response, "text"):
            st.markdown(response.text)
        else:
            st.markdown(response)




                
# run_nl_query(question, df)
with col3:
    run_button_3 = st.button("Generate Report using AI")
    
if run_button_3:
        if df is None:
            st.warning("Please upload or load a dataset first.")
        else:
            with st.spinner("Running AI..."):
                # time.sleep(5)  # remove this in production
                if gemini_api_key:
                    response = run_nl_query(auto_report_using_ai_prompt(), df, gemini_api_key)
                else:
                    st.warning('To enable AI Features, you must pass API-KEY')
                    response = None
                # st.write(response)

with col4:
    if st.button("Chat with data🤖"):
        st.session_state.chat_mode = True

# Initialize state
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = False

# Show chat UI AFTER button click
if st.session_state.chat_mode:
    if df is None:
        st.warning("Please upload or load a dataset first.")
    else:
        st.markdown("<br>",unsafe_allow_html=True) 
        st.header('🤖🤖 ASK AI 👇👇')
        st.markdown("---") 
        for i,j in enumerate(generate_suggestions(df)):
            st.markdown(f'**Q{i+1}:{j}**')
        chat = st.chat_input("Ask Gini👦:")

        if chat:
            with st.spinner("Running AI..."):
                if gemini_api_key:
                    response = run_nl_query(chat_prompt(df, chat), df, gemini_api_key)
                else:
                    st.warning('To enable AI Features, you must pass API-KEY')
                    response = None

st.markdown("""
<hr>

<p align="center">
  Connect with me:
</p>

<p align="center">
  <a href="https://github.com/your-username" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="40"/>
  </a>
  
  <a href="https://linkedin.com/in/your-profile" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="40"/>
  </a>
  
  <a href="https://twitter.com/your-handle" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/733/733579.png" width="40"/>
  </a>
</p>

<p align="center">
  Made with ❤️ using Streamlit
</p>
""", unsafe_allow_html=True)

from google import genai
import getpass
from prompt import text_prompt
import streamlit as st
import io, sys, re, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# api = getpass.getpass('Enter API key: ')
# client = genai.Client(api_key = api)

def summary_using_ai(df, api):
    client = genai.Client(api_key = api)
    
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text_prompt(df.columns,
                             df.sample(5),
                            df.describe(),
                            df.corr(numeric_only=True)),
    )
    return response



def run_nl_query(question, df, api):

    prompt = f"""
    You are a Python data analyst.

    Convert the user query into executable Python code using pandas.

    DataFrame name: df

    User Query:
    {question}

    Rules:
    - Output ONLY valid Python code (no markdown, no explanation)
    - Use dataframe 'df'
    - Import numpy as np if needed
    - Store final answer in variable 'result'

    🔥 OUTPUT RULES:
    -- Show One markdown at the top that tells about the dataset and which sector it belongs to and what it represents
    - NEVER return JSON or dictionary
    - NEVER use json.dumps()
    - For tables → return pandas DataFrame/Series
    - For insights → return markdown formatted string
    - If multiple outputs:
        → prioritize a clean markdown summary
        → OR return most important dataframe

    🔥 VISUALIZATION RULES:
    - Use matplotlib/seaborn
    - Always create:
        fig, ax = plt.subplots()
    - Assign:
        result = fig
    - Do NOT use plt.show()

    🔥 STYLE RULES:
    - Use palettes: 'Blues', 'Greys', 'viridis', 'tab10'
    - fig facecolor = '#ffffff'
    - text color = '#111827'

    🔥 MATPLOTLIB RULES:
    - Do NOT use 'ha' or 'va' in ticks
    - Use rotation=45
    - Align labels using:
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

    - No print()
    - No file/system operations

    Columns: {list(df.columns)}
    """

    # 🔐 Generate code
    client = genai.Client(api_key=api)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    code = response.text.strip()
    code = re.sub(r"```python|```", "", code).strip()
    code = re.sub(r"plt\.show\(\)", "", code)

    # 🚫 Basic safety check
    danger = ["import os", "import sys", "subprocess", "__import__", "open(", "exec(", "eval("]
    if any(d in code for d in danger):
        st.error("🚫 Unsafe code detected — blocked.")
        return None

    # ✅ Show generated code (typing effect FIXED)
    st.markdown("### 🧠 Generated Code")
    placeholder = st.empty()
    displayed_code = ""

    for line in code.split("\n"):
        displayed_code += line + "\n"
        placeholder.code(displayed_code, language="python")
        time.sleep(0.01)

    # 🔧 Execution environment
    local_vars = {
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns
    }

    try:
        # Reset plots
        plt.close('all')

        # Capture prints (just in case)
        buffer = io.StringIO()
        sys.stdout = buffer

        exec(code, local_vars)

        sys.stdout = sys.__stdout__

        printed_output = buffer.getvalue()
        result = local_vars.get("result", None)

        # ✅ Get ALL figures (FIX for multiple visuals)
        figs = [plt.figure(i) for i in plt.get_fignums()]

        # =========================
        # 🎯 DISPLAY SECTION
        # =========================

        if printed_output:
            st.markdown("### 🖨️ Output")
            st.text(printed_output)

        if result is not None:
            st.markdown("### 📊 Result")

            # 🔥 HANDLE DICT (YOUR MAIN ISSUE)
            if isinstance(result, dict):
                for key, value in result.items():

                    st.markdown(f"### 📌 {key.replace('_',' ').title()}")

                    # Try converting string tables
                    if isinstance(value, str):
                        try:
                            df_temp = pd.read_csv(io.StringIO(value))
                            st.dataframe(df_temp, use_container_width=True)
                        except:
                            st.markdown(value)
                    
                    elif isinstance(value, pd.DataFrame):
                        st.dataframe(value, use_container_width=True)

                    elif isinstance(value, pd.Series):
                        st.dataframe(value, use_container_width=True)

                    elif isinstance(value, dict):
                        st.json(value)

                    else:
                        st.write(value)

            elif isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)

            elif isinstance(result, pd.Series):
                st.dataframe(result, use_container_width=True)

            elif isinstance(result, str):
                st.markdown(result)

            else:
                st.write(result)

        # ✅ SHOW ALL VISUALS (FIXED 🔥)
        if figs:
            st.markdown("### 📈 Visualizations")
            for fig in figs:
                fig.patch.set_facecolor('#ffffff')
                st.pyplot(fig)

        elif result is None:
            try:
                st.pyplot(plt.gcf())
            except:
                st.info("No output returned.")

        plt.close('all')

        return result

    except Exception as e:
        sys.stdout = sys.__stdout__
        st.error(f"❌ Execution Error: {e}")
        return None




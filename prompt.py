def text_prompt(columns, sample_data, describe_data, corr_data):
    text = f'''You are a senior data analyst performing Automated Exploratory Data Analysis (EDA).

        I will provide:
        1. Dataset sample (first few rows)
        2. Summary statistics (describe())
        3. Column names
        4. Correlation matrix
        
        Your task is to generate a clear, structured, and concise EDA report.
        
        --- INPUT DATA ---
        
        Columns:
        {columns}
        
        Sample Data:
        {sample_data}
        
        Summary Statistics:
        {describe_data}
        
        Correlation Matrix:
        {corr_data}
        
        --- INSTRUCTIONS ---
        
        Perform the following analysis:
        
        1. **Dataset Overview**
           - Number of features
           - Likely data types (numerical, categorical, datetime if inferable)
           - General structure observations
        
        2. **Data Quality Checks**
           - Missing values (infer from sample/describe)
           - Potential duplicates (if detectable)
           - Outliers (based on min/max, quartiles)
           - Inconsistent or suspicious values
        
        3. **Feature-wise Insights**
           - Numerical features: distribution, skewness, spread
           - Categorical features: unique values, possible imbalance
           - Identify constant or near-constant columns
        
        4. **Correlation Analysis**
           - Strong positive/negative correlations (>|0.7|)
           - Multicollinearity risks
           - Interesting feature relationships
        
        5. **Key Patterns & Observations**
           - Trends, anomalies, or unusual behavior
           - Business-relevant insights (if inferable)
        
        6. **Feature Engineering Suggestions**
           - Encoding strategies
           - Scaling/normalization
           - Feature creation ideas
        
        7. **Recommendations**
           - Data cleaning steps
           - Columns to drop/keep
           - Next steps (modeling, visualization ideas)
        
        --- OUTPUT FORMAT ---
        
        Return output in clean markdown with headings:
        
        # 📊 Auto EDA Report
        
        ## Overview
        ## Data Quality Issues
        ## Feature Insights
        ## Correlation Insights
        ## Key Observations
        ## Feature Engineering Suggestions
        ## Recommendations
        
        Keep it concise but insightful. Avoid repeating raw data. Focus on actionable insights.'''
    return text


def auto_report_using_ai_prompt():
    text = '''You are an expert Data Analyst and Python developer.

        Your task is to perform a COMPLETE Exploratory Data Analysis (EDA) on the dataset using Python.
        
        You will be given:
        - DataFrame name: df
        - Sample data
        - Column names
        - Summary statistics (describe)
        - Correlation matrix (if available)
        - use group by for bi-variate and multivariate analysis
        Do all Univariate , bivariate and Multivariate Analysis whatever possible and show plots also
        ## 1. Univariate Analysis
                                    - For numerical columns:
                                      - Distribution plots (histplot / kdeplot)
                                      - Boxplots (for outlier detection)
                                    - For categorical columns:
                                      - Count plots
                                      - Value counts summary
                                    
                                    ## 2. Bivariate Analysis
                                    - Numerical vs Numerical:
          - Scatter plots
          - Regression plots (if meaningful)
        - Categorical vs Numerical:
          - Boxplots / violin plots
        - Categorical vs Categorical:
          - Crosstab / heatmap
        
        ## 3. Multivariate Analysis
        - Pairplot (if dataset size allows)
        - Correlation heatmap
        - Grouped aggregations (groupby with multiple columns)
        
        ## 4. Data Quality Checks
        - Missing values visualization
        - Outlier detection (IQR or boxplot-based)
        
        ## 5. Feature Insights
        - Highlight important patterns using aggregations
        
        '''
    # text = '''You are an expert Data Analyst. Perform full EDA on df:

    #     Univariate: numeric (hist/KDE, box), categorical (count, value counts)
    #     Bivariate: numeric vs numeric (scatter/regression), categorical vs numeric (box/violin), categorical vs categorical (crosstab/heatmap)
    #     Multivariate: pairplot, correlation heatmap, groupby aggregations
    #     Data quality: missing values, outliers
    #     Insights: key patterns via aggregations
    #     '''
    return text

def chat_prompt(df, question):
    text = f"""
    You are an expert Python Data Analyst and Data Storyteller.

    Your job is to:
    1. Understand the dataset
    2. Answer the user query
    3. Generate insights
    4. Create visualizations when useful

    DataFrame name: df

    User Query:
    {question}

    =========================
    🧠 DATA UNDERSTANDING
    =========================
    - convert user query to pandas code and visuals code
    - First, understand what this dataset represents
    - Identify domain (e.g., restaurant, finance, sales, healthcare, etc.)
    - Identify key important columns

    =========================
    ⚡ OUTPUT RULES
    =========================
    - Output ONLY valid Python pandas code
    - Do NOT use markdown syntax like ``` 
    - Do NOT explain anything outside code
    - Final output MUST be stored in variable: result

    =========================
    📊 RESULT FORMAT
    =========================
    - Always start with a CLEAN markdown summary stored in result (string)

    The markdown must include:
    - Dataset description (what it represents)
    - Sector/domain
    - What the user asked
    - Key insights (bullet points)

    Example:
    result = f\"\"\"
    ## 📊 Dataset Overview
    This dataset belongs to ...

    ## ❓ User Question
    ...

    ## 🔍 Key Insights
    - Insight 1
    - Insight 2
    \"\"\"

    =========================
    📈 VISUALIZATION RULES (VERY IMPORTANT)
    =========================
    - Create visual ONLY if it adds value
    - Use matplotlib or seaborn

    - ALWAYS:
        fig, ax = plt.subplots(figsize=(8,5))

    - Assign:
        result = result   (keep markdown as primary)
        # DO NOT overwrite result with fig

    - Do NOT use plt.show()

    - Preferred charts:
        - Categorical → barplot (top 10)
        - Numerical → histplot / boxplot
        - Comparison → barplot
        - Relationship → scatterplot

    =========================
    🎨 STYLE RULES
    =========================
    - Use palettes: 'Blues', 'Greys', 'viridis', 'tab10'
    - Keep background white
    - Title color = '#111827'
    - Rotate x labels 45 degrees
    - Align right:
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

    =========================
    🚫 STRICT RULES
    =========================
    - NO print()
    - NO JSON
    - NO dict as final result
    - NO file operations
    - NO os/system usage
    - NO eval/exec inside generated code

    =========================
    📌 DATA INFO
    =========================
    Columns: {list(df.columns)}

    Data Types:
    {df.dtypes}

    """
    return text
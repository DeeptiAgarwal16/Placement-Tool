from io import BytesIO
import streamlit as st
import pandas as pd
import json
import time
from gemini_helper import GeminiQueryProcessor
from data_processor import DataProcessor

# Set page config
st.set_page_config(
    page_title="Placement Query AI",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .css-1d391kg { padding-top: 2rem; }  /* Reduce top padding */
        .stButton>button { width: 100%; }   /* Make buttons full-width */
        .stDataFrame { border-radius: 10px; } /* Round DataFrame edges */
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'gemini' not in st.session_state:
    st.session_state.gemini = GeminiQueryProcessor()
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_view' not in st.session_state:
    st.session_state.current_view = "table"
if 'query_results' not in st.session_state:
    st.session_state.query_results = None
if 'insights' not in st.session_state:
    st.session_state.insights = None

# Title with subtitle
st.markdown("<h1 style='text-align: center;'>Placement Query AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Upload your placement data Excel file and ask questions in natural language.</p>", unsafe_allow_html=True)

# Sidebar: File Upload
with st.sidebar:
    st.header("📂 Upload Placement Data")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.session_state.processor = DataProcessor(df)
            st.success("✅ File uploaded successfully!")

            # Show data summary
            with st.expander("📊 Data Preview"):
                st.write(f"**Rows:** {len(df)}, **Columns:** {len(df.columns)}")
                st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    # Display recent queries
    if len(st.session_state.query_history) > 0:
        with st.expander("🕵 Recent Queries"):
            for query in st.session_state.query_history[-5:]:
                st.markdown(f"- {query}")

# Updated sample queries to match required query formats
sample_queries = [
    "Fetch the records of section 'A' who have a package above 10 LPA",
    "Compare placement record of Section A and Section B based on various parameters",
    "Please correlate and plot between percentage of 10th, 12th and B.Tech and placement status",
    "Based on placement statistics, why are students not getting placed and where is scope for improvement?",
    "Show students with lowest CGPA who still got placed",
    "What factors contribute most to higher placement packages?"
]

# Query input with suggestions
st.subheader("Ask a Query")
query = st.text_input("Enter your query", key="query_input")

col1, col2 = st.columns([3, 1])
with col1:
    if query:
        filtered_suggestions = [s for s in sample_queries if query.lower() in s.lower()]
        if filtered_suggestions:
            selected_suggestion = st.selectbox("💡 Suggested Queries", filtered_suggestions, index=0)
            query = selected_suggestion  # Auto-fill query input

with col2:
    run_query = st.button("Run Query", use_container_width=True)

# Run Query Processing
if run_query and query:
    st.session_state.query_history.append(query)

    with st.spinner("⏳ Processing your query..."):
        try:
            df_info = st.session_state.processor.get_dataframe_info()
            query_instructions_table = st.session_state.gemini.process_query(query, df_info)

            instructions_table = json.loads(query_instructions_table)
            instructions_table["output_type"] = "table"
            table_result = st.session_state.processor.execute_query(json.dumps(instructions_table))

            instructions_graph = json.loads(query_instructions_table)
            instructions_graph["output_type"] = "graph"
            graph_result = st.session_state.processor.execute_query(json.dumps(instructions_graph))

            # Get insights for subjective questions
            is_subjective = any(keyword in query.lower() for keyword in ["why", "scope", "improvement", "factor", "contribute", "insight", "reason"])
            
            if is_subjective:
                insights = st.session_state.gemini.generate_insights(query, df_info, table_result)
                st.session_state.insights = insights
            else:
                st.session_state.insights = None

            st.session_state.query_results = {
                "table": table_result,
                "graph": graph_result,
                "query": query
            }

            st.session_state.current_view = "table"

        except Exception as e:
            st.error(f"🚨 Unexpected error: {str(e)}")

# Display Results in Tabs
if st.session_state.query_results:
    st.divider()
    
    # Add new tab for insights if available
    if st.session_state.insights:
        tab1, tab2, tab3 = st.tabs(["📊 Table View", "📈 Graph View", "🔍 Insights"])
    else:
        tab1, tab2 = st.tabs(["📊 Table View", "📈 Graph View"])

    with tab1:
        table_result = st.session_state.query_results["table"]
        if table_result["type"] == "error":
            st.error(table_result["message"])
        else:
            st.subheader(f"📊 {table_result['title']}")
            st.dataframe(table_result["data"])
            
            # Download as Excel button
            output = BytesIO()
            pd.DataFrame(table_result["data"]).to_excel(output, index=False, engine="xlsxwriter")
            output.seek(0)
            st.download_button(label="📥 Download as Excel", data=output, file_name="query_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab2:
        graph_result = st.session_state.query_results["graph"]
        if graph_result["type"] == "error":
            st.error(graph_result["message"])
        else:
            st.subheader(f"📈 {graph_result['title']}")
            st.plotly_chart(graph_result["figure"], use_container_width=True)
    
    # Show insights tab if available
    if st.session_state.insights:
        with tab3:
            st.subheader("🔍 AI Insights")
            st.markdown(st.session_state.insights)

else:
    st.info("📌 Please upload an Excel file to start querying.")
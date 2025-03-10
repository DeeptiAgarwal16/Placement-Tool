# import streamlit as st
# import pandas as pd
# import json
# from gemini_helper import GeminiQueryProcessor
# from data_processor import DataProcessor

# # Set page config
# st.set_page_config(
#     page_title="Placement Data Query Tool",
#     page_icon="🎓",
#     layout="wide"
# )

# # Initialize session state
# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'processor' not in st.session_state:
#     st.session_state.processor = None
# if 'gemini' not in st.session_state:
#     st.session_state.gemini = GeminiQueryProcessor()
# if 'query_history' not in st.session_state:
#     st.session_state.query_history = []

# # Title
# st.title("College Placement Data Query Tool")
# st.markdown("Upload your placement data Excel file and ask questions in natural language")

# # Sidebar for file upload
# with st.sidebar:
#     st.header("Upload Data")
#     uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
    
#     if uploaded_file is not None:
#         try:
#             df = pd.read_excel(uploaded_file)
#             st.session_state.df = df
#             st.session_state.processor = DataProcessor(df)
#             st.success("File uploaded successfully!")
            
#             # Display dataframe info
#             st.subheader("Data Summary")
#             st.write(f"Rows: {len(df)}")
#             st.write(f"Columns: {', '.join(df.columns)}")
            
#         except Exception as e:
#             st.error(f"Error loading file: {str(e)}")
    
#     # Display query history
#     if len(st.session_state.query_history) > 0:
#         st.subheader("Recent Queries")
#         for query in st.session_state.query_history[-5:]:
#             st.markdown(f"- {query}")

# # Main content area
# if st.session_state.df is not None:
#     # Query input
#     query = st.text_input("Enter your query", placeholder="e.g., Show me students who are not placed yet")
    
#     if query and st.button("Run Query"):
#         st.session_state.query_history.append(query)
        
#         with st.spinner("Processing your query..."):
#             # Get dataframe info for Gemini
#             df_info = st.session_state.processor.get_dataframe_info()
            
#             # Process query with Gemini
#             try:
#                 query_instructions = st.session_state.gemini.process_query(query, df_info)
#                 st.session_state.last_instructions = query_instructions
                
#                 # Execute query
#                 result = st.session_state.processor.execute_query(query_instructions)
                
#                 if result["type"] == "error":
#                     st.error(result["message"])
#                 else:
#                     # Display results
#                     st.subheader(result["title"])
                    
#                     if result["type"] == "table":
#                         st.dataframe(result["data"])
                        
#                         # Download button for Excel
#                         excel_file = st.session_state.processor.generate_excel(result["data"])
#                         st.download_button(
#                             label="Download as Excel",
#                             data=excel_file,
#                             file_name="query_result.xlsx",
#                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#                         )
                        
#                         # Get explanation
#                         result_summary = f"Table with {len(result['data'])} rows and {len(result['data'].columns)} columns."
#                         explanation = st.session_state.gemini.explain_result(query, result_summary)
#                         st.info(explanation)
                        
#                     elif result["type"] == "graph":
#                         st.plotly_chart(result["figure"], use_container_width=True)
                        
#                         # Get explanation
#                         explanation = st.session_state.gemini.explain_result(query, f"Graph showing {result['title']}")
#                         st.info(explanation)
                        
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")
    
#     # Sample queries suggestions
#     st.subheader("Sample Queries")
#     sample_queries = [
#         "Show all students who are not placed yet",
#         "Compare placement percentages between Section A and Section B",
#         "Show me the top 5 companies by number of placements",
#         "What's the average salary of placed students by department?",
#         "How many students got placed with salary above 10 LPA?"
#     ]
    
#     cols = st.columns(3)
#     for i, sample in enumerate(sample_queries):
#         col_index = i % 3
#         if cols[col_index].button(sample):
#             # Set the query input and simulate button click
#             st.experimental_rerun()  # This will rerun the app in a hacky way to simulate a form submission
# else:
#     st.info("Please upload an Excel file to get started")
    
#     # Sample data format
#     st.subheader("Expected data format")
#     st.markdown("""
#     Your Excel file should contain columns such as:
#     - Student ID
#     - Name
#     - Section
#     - Department
#     - CGPA
#     - Placement Status (e.g., 'Placed', 'Not Placed')
#     - Company (if placed)
#     - Salary (if placed)
#     - etc.
#     """)
    
#     # Example dataframe
#     example_data = {
#         'Student ID': ['S001', 'S002', 'S003', 'S004'],
#         'Name': ['Alex Smith', 'Bailey Johnson', 'Casey Williams', 'Dana Brown'],
#         'Section': ['A', 'B', 'A', 'B'],
#         'Department': ['CSE', 'IT', 'ECE', 'CSE'],
#         'CGPA': [8.9, 7.6, 9.2, 8.1],
#         'Placement Status': ['Placed', 'Not Placed', 'Placed', 'Not Placed'],
#         'Company': ['TechCorp', None, 'DataSys', None],
#         'Salary (LPA)': [12.5, None, 14.0, None]
#     }
    
#     st.dataframe(pd.DataFrame(example_data))




from io import BytesIO
import streamlit as st
import pandas as pd
import json
import time
from gemini_helper import GeminiQueryProcessor
from data_processor import DataProcessor

# Set page config
st.set_page_config(
    page_title="Placement Data Query Tool",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'gemini' not in st.session_state:
    st.session_state.gemini = GeminiQueryProcessor()
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Title
st.title("College Placement Data Query Tool")
st.markdown("Upload your placement data Excel file and ask questions in natural language")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.session_state.processor = DataProcessor(df)
            st.success("File uploaded successfully!")
            
            # Display dataframe info
            st.subheader("Data Summary")
            st.write(f"Rows: {len(df)}")
            st.write(f"Columns: {', '.join(df.columns)}")
            
            # Show sample data
            with st.expander("View sample data"):
                st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Display query history
    if len(st.session_state.query_history) > 0:
        st.subheader("Recent Queries")
        for query in st.session_state.query_history[-5:]:
            st.markdown(f"- {query}")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)

# Main content area
if st.session_state.df is not None:
    # Query input
    query = st.text_input("Enter your query", placeholder="e.g., Show me students who are not placed yet")
    
    # Add examples to help the user
    with st.expander("Example queries"):
        st.markdown("""
        - Show all students who are not placed yet
        - Compare placement percentages between Section A and Section B
        - Show me the top 5 companies by number of placements
        - What's the average salary of placed students by department?
        - How many students got placed with salary above 10 LPA?
        """)
    
    if query and st.button("Run Query"):
        st.session_state.query_history.append(query)
        
        with st.spinner("Processing your query..."):
            try:
                # Get dataframe info for Gemini
                df_info = st.session_state.processor.get_dataframe_info()
                
                # Process query with Gemini - add a small delay to ensure API is responsive
                time.sleep(0.5)
                query_instructions = st.session_state.gemini.process_query(query, df_info)
                
                # Show the raw JSON in debug mode
                if st.session_state.debug_mode:
                    st.subheader("Debug: Raw JSON from Gemini")
                    st.code(query_instructions, language="json")
                
                # Execute query
                result = st.session_state.processor.execute_query(query_instructions)
                
                if result["type"] == "error":
                    st.error(result["message"])
                else:
                    # Display results
                    st.subheader(result["title"])
                    
                    if result["type"] == "table":
                        # Show the number of rows returned
                        st.write(f"Found {len(result['data'])} results")
                        
                        # Display the table
                        st.dataframe(result["data"])
                        
                        # Download button for Excel
                        excel_file = st.session_state.processor.generate_excel(result["data"])
                        st.download_button(
                            label="Download as Excel",
                            data=excel_file,
                            file_name="query_result.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        # Get explanation
                        result_summary = f"Table with {len(result['data'])} rows and {len(result['data'].columns)} columns."
                        explanation = st.session_state.gemini.explain_result(query, result_summary)
                        st.info(explanation)
                        
                    elif result["type"] == "graph":
                        st.plotly_chart(result["figure"], use_container_width=True)
                        
                        # Get explanation
                        explanation = st.session_state.gemini.explain_result(query, f"Graph showing {result['title']}")
                        st.info(explanation)
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
    
    # Sample queries suggestions
    st.subheader("Sample Queries")
    sample_queries = [
        "Show all students who are not placed yet",
        "Compare placement percentages between Section A and Section B",
        "Show me the top 5 companies by number of placements",
        "What's the average salary of placed students by department?",
        "How many students got placed with salary above 10 LPA?"
    ]
    
    cols = st.columns(3)
    for i, sample in enumerate(sample_queries):
        col_index = i % 3
        if cols[col_index].button(sample):
            # We have to use a workaround since we can't directly set the value of text_input
            st.session_state.last_sample = sample
            st.experimental_rerun()
else:
    st.info("Please upload an Excel file to get started")
    
    # Sample data format
    st.subheader("Expected data format")
    st.markdown("""
    Your Excel file should contain columns such as:
    - Student ID
    - Name
    - Section
    - Department
    - CGPA
    - Placement Status (e.g., 'Placed', 'Not Placed')
    - Company (if placed)
    - Salary (if placed)
    - etc.
    """)
    
    # Example dataframe
    example_data = {
        'Student ID': ['S001', 'S002', 'S003', 'S004'],
        'Name': ['Alex Smith', 'Bailey Johnson', 'Casey Williams', 'Dana Brown'],
        'Section': ['A', 'B', 'A', 'B'],
        'Department': ['CSE', 'IT', 'ECE', 'CSE'],
        'CGPA': [8.9, 7.6, 9.2, 8.1],
        'Placement Status': ['Placed', 'Not Placed', 'Placed', 'Not Placed'],
        'Company': ['TechCorp', None, 'DataSys', None],
        'Salary (LPA)': [12.5, None, 14.0, None]
    }
    
    st.dataframe(pd.DataFrame(example_data))
    
    # Download example template
    example_df = pd.DataFrame(example_data)
    
    @st.cache_data
    def get_example_excel():
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            example_df.to_excel(writer, index=False, sheet_name='Example Data')
        output.seek(0)
        return output
    
    st.download_button(
        label="Download Example Template",
        data=get_example_excel(),
        file_name="placement_data_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
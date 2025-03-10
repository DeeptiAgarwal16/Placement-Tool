import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiQueryProcessor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    def process_query(self, query, dataframe_info):
        """
        Process natural language query using Gemini API
        
        Args:
            query (str): Natural language query about placement data
            dataframe_info (str): Information about the dataframe structure
            
        Returns:
            dict: Processing instructions for the query
        """
        prompt = f"""
        You are a data analysis assistant that converts natural language queries into Python pandas operations.
        
        Here's information about the dataframe:
        {dataframe_info}
        
        The user's query is: "{query}"
        
        Your task is to generate Python code to answer this query. The code should:
        1. Use 'df' as the variable name for the dataframe
        2. Use correct pandas functions and methods
        3. Make sure final results are stored in a variable called 'result_df'
        
        Respond with a JSON object that contains:
        1. "query_type": Either "filter", "comparison", or "analysis"
        2. "operations": List of Python code lines as strings. Each line should be syntactically valid and executable. Do NOT include assignment like 'result = {{...}}' without proper Python syntax.
        3. "output_type": Either "table" or "graph"
        4. "graph_type" (if output_type is graph): Type of graph to generate (bar, line, pie, etc.)
        5. "title": Suggested title for the result
        
        For comparison queries between categories (like sections A and B), make sure to prepare the data correctly for visualization.
        
        Format your response as valid, parseable JSON with no additional text or code blocks.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Try to parse the JSON to validate it
            try:
                json_data = json.loads(response_text)
                
                # Validate and fix the operations if needed
                if "operations" in json_data:
                    # Make sure the last operation assigns to result_df if it doesn't already
                    operations = json_data["operations"]
                    if operations and not any("result_df" in op for op in operations):
                        # Check if there's any result variable
                        result_vars = [op.split("=")[0].strip() for op in operations if "=" in op]
                        if result_vars:
                            last_result = result_vars[-1]
                            operations.append(f"result_df = {last_result}")
                        else:
                            # Just assign the last expression to result_df
                            last_op = operations[-1]
                            if "=" not in last_op and not last_op.strip().startswith("result_df"):
                                operations[-1] = f"result_df = {last_op}"
                    
                    json_data["operations"] = operations
                    return json.dumps(json_data)
                
                return response_text
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from the text
                # Sometimes LLMs add additional text before/after the JSON
                import re
                json_pattern = r'```json\s*([\s\S]*?)\s*```|^\s*(\{[\s\S]*\})\s*$'
                match = re.search(json_pattern, response_text)
                
                if match:
                    json_str = match.group(1) or match.group(2)
                    # Validate it's parseable
                    json_data = json.loads(json_str)
                    
                    # Apply the same fixes to operations
                    if "operations" in json_data:
                        operations = json_data["operations"]
                        if operations and not any("result_df" in op for op in operations):
                            result_vars = [op.split("=")[0].strip() for op in operations if "=" in op]
                            if result_vars:
                                last_result = result_vars[-1]
                                operations.append(f"result_df = {last_result}")
                            else:
                                last_op = operations[-1]
                                if "=" not in last_op and not last_op.strip().startswith("result_df"):
                                    operations[-1] = f"result_df = {last_op}"
                        
                        json_data["operations"] = operations
                    
                    return json.dumps(json_data)
                else:
                    # Create a fallback JSON with an error message
                    fallback = {
                        "query_type": "filter",
                        "operations": ["df = self.df", "result_df = df.head()"],
                        "output_type": "table",
                        "title": "Error processing query - showing sample data"
                    }
                    return json.dumps(fallback)
                    
        except Exception as e:
            # Return a fallback JSON if something goes wrong
            fallback = {
                "query_type": "filter",
                "operations": ["df = self.df", "result_df = df.head()"],
                "output_type": "table",
                "title": f"Error: {str(e)}"
            }
            return json.dumps(fallback)
        
    def explain_result(self, query, result_summary):
        """
        Generate a natural language explanation of the results
        
        Args:
            query (str): Original natural language query
            result_summary (str): Summary of the results
            
        Returns:
            str: Natural language explanation of the results
        """
        prompt = f"""
        The user asked: "{query}"
        
        Based on the data analysis, here are the results:
        {result_summary}
        
        Provide a clear, concise explanation of these results in 2-3 sentences.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"




# import os
# import json
# import google.generativeai as genai
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Configure Gemini API
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# class GeminiQueryProcessor:
#     def __init__(self):
#         self.model = genai.GenerativeModel('gemini-1.5-pro')
        
#     def process_query(self, query, dataframe_info):
#         """
#         Process natural language query using Gemini API
        
#         Args:
#             query (str): Natural language query about placement data
#             dataframe_info (str): Information about the dataframe structure
            
#         Returns:
#             dict: Processing instructions for the query
#         """
#         prompt = f"""
#         You are a data analysis assistant that converts natural language queries into Python pandas operations.
        
#         Here's information about the dataframe:
#         {dataframe_info}
        
#         The user's query is: "{query}"
        
#         Respond with a JSON object that contains:
#         1. "query_type": Either "filter", "comparison", or "analysis"
#         2. "operations": List of pandas operations to perform (as Python code strings that can be executed using the variable 'self.df' as the dataframe)
#         3. "output_type": Either "table" or "graph"
#         4. "graph_type" (if output_type is graph): Type of graph to generate (bar, line, pie, etc.)
#         5. "title": Suggested title for the result
        
#         Format your response as valid, parseable JSON with no additional text. Ensure all Python code in the 'operations' field is properly escaped for JSON.
#         """
        
#         try:
#             response = self.model.generate_content(prompt)
#             response_text = response.text
            
#             # Try to parse the JSON to validate it
#             try:
#                 json_data = json.loads(response_text)
#                 return response_text
#             except json.JSONDecodeError:
#                 # If not valid JSON, try to extract JSON from the text
#                 # Sometimes LLMs add additional text before/after the JSON
#                 import re
#                 json_pattern = r'```json\s*([\s\S]*?)\s*```|^\s*(\{[\s\S]*\})\s*$'
#                 match = re.search(json_pattern, response_text)
                
#                 if match:
#                     json_str = match.group(1) or match.group(2)
#                     # Validate it's parseable
#                     json.loads(json_str)
#                     return json_str
#                 else:
#                     # Create a fallback JSON with an error message
#                     fallback = {
#                         "query_type": "filter",
#                         "operations": ["self.df.head()"],
#                         "output_type": "table",
#                         "title": "Error processing query - showing sample data"
#                     }
#                     return json.dumps(fallback)
                    
#         except Exception as e:
#             # Return a fallback JSON if something goes wrong
#             fallback = {
#                 "query_type": "filter",
#                 "operations": ["self.df.head()"],
#                 "output_type": "table",
#                 "title": f"Error: {str(e)}"
#             }
#             return json.dumps(fallback)
        
#     def explain_result(self, query, result_summary):
#         """
#         Generate a natural language explanation of the results
        
#         Args:
#             query (str): Original natural language query
#             result_summary (str): Summary of the results
            
#         Returns:
#             str: Natural language explanation of the results
#         """
#         prompt = f"""
#         The user asked: "{query}"
        
#         Based on the data analysis, here are the results:
#         {result_summary}
        
#         Provide a clear, concise explanation of these results in 2-3 sentences.
#         """
        
#         try:
#             response = self.model.generate_content(prompt)
#             return response.text
#         except Exception as e:
#             return f"Unable to generate explanation: {str(e)}"
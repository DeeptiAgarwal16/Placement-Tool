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
        # Categorize the query type
        query_category = self._categorize_query(query)
        
        # Build prompt based on query type
        prompt = self._build_prompt_for_category(query, dataframe_info, query_category)
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Try to parse the JSON to validate it
            try:
                json_data = json.loads(response_text)
                json_data = self._post_process_json(json_data)
                return json.dumps(json_data)
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
                    json_data = self._post_process_json(json_data)
                    return json.dumps(json_data)
                else:
                    # Create a fallback JSON with an error message
                    return self._create_fallback_response("Could not parse response as JSON")
                    
        except Exception as e:
            # Return a fallback JSON if something goes wrong
            return self._create_fallback_response(f"Error: {str(e)}")
    
    def _categorize_query(self, query):
        """
        Categorize the query into one of several types
        
        Args:
            query (str): The natural language query
            
        Returns:
            str: Query category
        """
        query = query.lower()
        
        # Section-specific filtering
        if ("section" in query and ("above" in query or "below" in query or "more than" in query or "less than" in query)):
            return "section_filter"
            
        # Comparison between sections
        if ("compare" in query or "comparison" in query) and "section" in query:
            return "section_comparison"
            
        # Correlation analysis
        if ("correlate" in query or "correlation" in query or "relationship between" in query or "plot between" in query):
            return "correlation"
            
        # Subjective analysis
        if ("why" in query or "improvement" in query or "suggest" in query or "recommendation" in query 
            or "factor" in query or "contribute" in query or "scope" in query):
            return "subjective"
            
        # Default to basic filter if no other category matches
        return "basic_filter"
    
    def _build_prompt_for_category(self, query, dataframe_info, category):
        """
        Build a specialized prompt based on query category
        
        Args:
            query (str): The natural language query
            dataframe_info (str): Information about the dataframe structure
            category (str): Query category
            
        Returns:
            str: Specialized prompt
        """
        base_prompt = f"""
        You are a data analysis assistant that converts natural language queries into Python pandas operations.
        
        Here's information about the dataframe:
        {dataframe_info}
        
        The user's query is: "{query}"
        """
        
        # Add specialized instructions based on category
        if category == "section_filter":
            base_prompt += """
            This is a section-specific filter query. Make sure to:
            1. Filter the dataframe correctly based on the section mentioned
            2. Apply the numerical filter on the package/LPA value
            3. Sort the results appropriately (usually by package in descending order)
            """
        
        elif category == "section_comparison":
            base_prompt += """
            This is a comparison query between different sections. Make sure to:
            1. Split the analysis by section
            2. Compare on multiple parameters:
               - Average package
               - Placement rate
               - Maximum and minimum packages
               - Distribution of packages (using appropriate binning)
            3. Prepare the data in a format suitable for multiple graphs (bar charts, grouped bar charts)
            4. Consider using plotly for visualizations
            5. Add statistical analysis like t-tests to determine if differences are significant
            """
        
        elif category == "correlation":
            base_prompt += """
            This is a correlation analysis query. Make sure to:
            1. Calculate correlation coefficients between academic parameters (10th, 12th, B.Tech percentages) and placement outcomes
            2. Prepare scatter plots or heatmaps to visualize correlations
            3. Consider calculating separate correlations for different groups (e.g., by section)
            4. Include code to create appropriate visualization of correlations
            5. Use pandas' corr() method and plotly visualization libraries appropriately
            """
        
        elif category == "subjective":
            base_prompt += """
            This is a subjective analysis query. Make sure to:
            1. Identify patterns in the data that might explain placement outcomes
            2. Look for potential areas of improvement based on the data
            3. Analyze the characteristics of placed vs. non-placed students
            4. Generate statistics and visualizations that help identify improvement areas
            5. Your code should prepare data that illustrates key insights
            """
        
        # Common instructions for all categories
        base_prompt += """
        
        Your task is to generate Python code to answer this query. The code should:
        1. Use 'df' as the variable name for the dataframe
        2. Use correct pandas functions and methods
        3. Make sure final results are stored in a variable called 'result_df'
        4. Include any necessary imports (pandas, numpy, matplotlib, seaborn, etc.)
        5. For visualizations, use plotly (px or go) as that's what the application supports
        
        Respond with a JSON object that contains:
        1. "query_type": One of "filter", "comparison", "correlation", "analysis"
        2. "operations": List of Python code lines as strings. Each line should be syntactically valid and executable.
        3. "output_type": One of "table", "graph", "combined"
        4. "graph_type" (if output_type includes graph): Type of graph to generate (bar, line, pie, scatter, heatmap, etc.)
        5. "title": Suggested title for the result
        6. "insights": 2-3 key insights that can be derived from this analysis
        
        For graphs, also add these whenever relevant:
        - "x_column": Name of column to use for x-axis
        - "y_column": Name of column to use for y-axis
        - "color_column": Name of column to use for coloring (if applicable)
        - "names_column": Name of column for pie chart names (if applicable)
        - "values_column": Name of column for pie chart values (if applicable)
        
        Format your response as valid, parseable JSON with no additional text or code blocks.
        """
        
        return base_prompt
    
    def _post_process_json(self, json_data):
        """
        Post-process the JSON response to ensure it's valid and complete
        
        Args:
            json_data (dict): The parsed JSON response
            
        Returns:
            dict: The processed JSON data
        """
        # Ensure operations list exists
        if "operations" not in json_data:
            json_data["operations"] = ["result_df = df.head()"]
            
        # Make sure the output assigns to result_df
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
        
        # Ensure other required fields exist
        if "query_type" not in json_data:
            json_data["query_type"] = "filter"
            
        if "output_type" not in json_data:
            json_data["output_type"] = "table"
            
        if "title" not in json_data:
            json_data["title"] = "Analysis Results"
            
        if "graph_type" not in json_data and json_data["output_type"] in ["graph", "combined"]:
            json_data["graph_type"] = "bar"
            
        if "insights" not in json_data:
            json_data["insights"] = ["Analysis complete", "Review the data for insights"]
        
        # Add import statements for common libraries if not present
        has_imports = any("import" in op for op in operations)
        if not has_imports:
            # Insert imports at the beginning
            operations.insert(0, "import pandas as pd")
            operations.insert(1, "import numpy as np")
            
            # Add plotly import if it's a graph
            if json_data["output_type"] in ["graph", "combined"]:
                operations.insert(2, "import plotly.express as px")
                operations.insert(3, "import plotly.graph_objects as go")
        
        json_data["operations"] = operations
        return json_data
    
    def _create_fallback_response(self, error_message):
        """
        Create a fallback JSON response for error cases
        
        Args:
            error_message (str): Description of the error
            
        Returns:
            str: JSON string with fallback response
        """
        fallback = {
            "query_type": "filter",
            "operations": ["import pandas as pd", "df = self.df", "result_df = df.head()"],
            "output_type": "table",
            "title": f"Error processing query: {error_message}",
            "insights": ["Error encountered during query processing", "Showing sample data instead"]
        }
        return json.dumps(fallback)
        
    def explain_result(self, query, result_summary, insights=None):
        """
        Generate a natural language explanation of the results
        
        Args:
            query (str): Original natural language query
            result_summary (str): Summary of the results
            insights (list, optional): Key insights from the analysis
            
        Returns:
            str: Natural language explanation of the results
        """
        insights_text = ""
        if insights and isinstance(insights, list):
            insights_text = "Key insights:\n- " + "\n- ".join(insights)
        
        prompt = f"""
        The user asked: "{query}"
        
        Based on the data analysis, here are the results:
        {result_summary}
        
        {insights_text}
        
        Provide a clear, concise explanation of these results in 3-5 sentences.
        If this was a comparison query, highlight the key differences found.
        If this was a correlation query, explain the strength and direction of correlations.
        If this was a subjective query, suggest actionable recommendations based on the data.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"
    
    def generate_insights(self, query, dataframe_info, result_data):
        """
        Generate insights and recommendations based on query results
        
        Args:
            query (str): Original natural language query
            dataframe_info (str): Information about the dataframe structure
            result_data (dict): Result data from query execution
            
        Returns:
            str: Markdown-formatted insights and recommendations
        """
        # For subjective queries, provide in-depth analysis and recommendations
        is_subjective = any(keyword in query.lower() for keyword in ["why", "scope", "improvement", "factor", "contribute", "insight", "reason"])
        
        # Convert result data to string representation
        if isinstance(result_data, dict) and "data" in result_data:
            if hasattr(result_data["data"], "to_markdown"):
                data_str = result_data["data"].to_markdown()
            else:
                data_str = str(result_data["data"])
        else:
            data_str = str(result_data)
            
        if is_subjective:
            prompt = f"""
            The user asked: "{query}"
            
            Here's information about the dataframe:
            {dataframe_info}
            
            Based on the analysis, we have these results:
            {data_str}
            
            Provide a comprehensive analysis with these sections:
            
            1. **Key Findings**: 3-4 major insights from the data analysis
            2. **Root Causes**: Identify potential reasons for the placement outcomes based on the data
            3. **Areas for Improvement**: 3-5 specific areas where improvements could lead to better placement outcomes
            4. **Actionable Recommendations**: 4-6 concrete, specific recommendations for:
               - Students trying to improve their placement prospects
               - Institution/department to better prepare students
               - Addressing skill gaps or other issues identified
            
            Format your response using Markdown with proper headings, bullet points, and emphasis where appropriate.
            Focus on being specific, data-driven, and actionable rather than generic advice.
            """
        else:
            # For regular queries, provide a simpler insight summary
            prompt = f"""
            The user asked: "{query}"
            
            Here's information about the dataframe:
            {dataframe_info}
            
            Based on the analysis, we have these results:
            {data_str}
            
            Provide a concise analysis with these sections:
            
            1. **Summary**: 2-3 sentence summary of the key findings
            2. **Key Insights**: 3-4 specific insights derived from this data
            3. **Implications**: What these findings might mean for students or the institution
            
            Format your response using Markdown with proper headings, bullet points, and emphasis where appropriate.
            """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"### Error Generating Insights\n\nUnable to generate insights: {str(e)}"
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import traceback

class DataProcessor:
    def __init__(self, df):
        """Initialize with the uploaded dataframe"""
        self.df = df
        
    def get_dataframe_info(self):
        """Get information about the dataframe structure"""
        info = {
            "columns": list(self.df.columns),
            "dtypes": {col: str(self.df[col].dtype) for col in self.df.columns},
            "sample": self.df.head(3).to_dict(orient="records"),
            "placement_status_values": list(self.df["Placement Status"].unique()) if "Placement Status" in self.df.columns else [],
            "section_values": list(self.df["Section"].unique()) if "Section" in self.df.columns else []
        }
        return json.dumps(info, indent=2)
    
    def execute_query(self, query_instructions):
        """Execute the query based on Gemini's instructions"""
        try:
            # Ensure we have valid JSON
            if not query_instructions or query_instructions.isspace():
                raise ValueError("Empty response from Gemini API")
                
            instructions = json.loads(query_instructions)
            
            query_type = instructions.get("query_type", "filter")
            operations = instructions.get("operations", ["self.df.head()"])
            output_type = instructions.get("output_type", "table")
            title = instructions.get("title", "Query Results")
            
            # Execute operations safely - with better handling for different operation formats
            result_df = None
            local_namespace = {'self': self, 'pd': pd, 'df': self.df}
            
            # Execute all operations in a single context
            combined_code = "\n".join(operations)
            
            # Add a line to capture the final result
            if not any(line.strip().startswith("result_df =") for line in operations):
                combined_code += "\nresult_df = " + operations[-1]
                
            # Execute the combined code in the local namespace
            exec(combined_code, {'pd': pd}, local_namespace)
            
            # Try to get the result from different possible variable names
            if 'result_df' in local_namespace:
                result_df = local_namespace['result_df']
            elif 'result' in local_namespace:
                result_df = local_namespace['result']
            else:
                # Fallback - try the last operation directly
                try:
                    result_df = eval(operations[-1], {'pd': pd}, local_namespace)
                except SyntaxError:
                    # If it's an assignment, just use the variable it likely created
                    var_name = operations[-1].split('=')[0].strip()
                    if var_name in local_namespace:
                        result_df = local_namespace[var_name]
                    else:
                        # Final fallback - show the processed dataframe
                        result_df = local_namespace.get('df', self.df)
            
            # Handle dictionary result - convert to DataFrame
            if isinstance(result_df, dict):
                result_df = pd.DataFrame([result_df])
            
            # Ensure we have a dataframe
            if not isinstance(result_df, pd.DataFrame):
                if isinstance(result_df, pd.Series):
                    result_df = result_df.to_frame()
                else:
                    # Handle various other result types
                    if isinstance(result_df, (int, float, str, bool)):
                        result_df = pd.DataFrame({'Result': [result_df]})
                    elif isinstance(result_df, list):
                        if all(isinstance(x, dict) for x in result_df):
                            result_df = pd.DataFrame(result_df)
                        else:
                            result_df = pd.DataFrame({'Result': result_df})
                    else:
                        # Last resort
                        result_df = pd.DataFrame({'Result': ['Unable to display result']})
            
            if output_type == "table":
                return {
                    "type": "table",
                    "data": result_df,
                    "title": title
                }
            elif output_type == "graph":
                graph_type = instructions.get("graph_type", "bar")
                fig = self.create_visualization(result_df, graph_type, title)
                return {
                    "type": "graph",
                    "figure": fig,
                    "title": title
                }
        except json.JSONDecodeError as e:
            return {
                "type": "error",
                "message": f"Invalid JSON response: {str(e)}\nResponse: {query_instructions[:100]}..."
            }
        except Exception as e:
            # Get the full traceback for debugging
            stack_trace = traceback.format_exc()
            return {
                "type": "error",
                "message": f"Error processing query: {str(e)}\n\nTraceback: {stack_trace}"
            }
    
    def create_visualization(self, df, graph_type, title):
        """Create visualization based on the type specified"""
        try:
            # Handle dictionary input for graph directly
            if isinstance(df, dict):
                if graph_type == "pie":
                    labels = list(df.keys())
                    values = list(df.values())
                    fig = px.pie(names=labels, values=values, title=title)
                    return fig
                else:
                    df = pd.DataFrame(list(df.items()), columns=['Category', 'Value'])
            
            # Regular dataframe visualization
            if graph_type == "bar":
                if len(df.columns) >= 2:
                    fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
                else:
                    fig = px.bar(df, title=title)
            elif graph_type == "line":
                if len(df.columns) >= 2:
                    fig = px.line(df, x=df.columns[0], y=df.columns[1], title=title)
                else:
                    fig = px.line(df, title=title)
            elif graph_type == "pie":
                if len(df.columns) >= 2:
                    fig = px.pie(df, names=df.columns[0], values=df.columns[1], title=title)
                else:
                    fig = px.pie(df, title=title)
            elif graph_type == "scatter":
                if len(df.columns) >= 3:
                    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=df.columns[2], title=title)
                elif len(df.columns) >= 2:
                    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title=title)
                else:
                    fig = px.scatter(df, title=title)
            else:
                # Default to bar chart
                fig = px.bar(df, title=title)
                
            return fig
        except Exception as e:
            # Create a simple figure with an error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def generate_excel(self, df):
        """Generate Excel file from dataframe"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        output.seek(0)
        return output


# import pandas as pd
# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from io import BytesIO
# import traceback

# class DataProcessor:
#     def __init__(self, df):
#         """Initialize with the uploaded dataframe"""
#         self.df = df
        
#     def get_dataframe_info(self):
#         """Get information about the dataframe structure"""
#         info = {
#             "columns": list(self.df.columns),
#             "dtypes": {col: str(self.df[col].dtype) for col in self.df.columns},
#             "sample": self.df.head(3).to_dict(orient="records"),
#             "placement_status_values": list(self.df["Placement Status"].unique()) if "Placement Status" in self.df.columns else [],
#             "section_values": list(self.df["Section"].unique()) if "Section" in self.df.columns else []
#         }
#         return json.dumps(info, indent=2)
    
#     def execute_query(self, query_instructions):
#         """Execute the query based on Gemini's instructions"""
#         try:
#             # Ensure we have valid JSON
#             if not query_instructions or query_instructions.isspace():
#                 raise ValueError("Empty response from Gemini API")
                
#             instructions = json.loads(query_instructions)
            
#             query_type = instructions.get("query_type", "filter")
#             operations = instructions.get("operations", ["self.df.head()"])
#             output_type = instructions.get("output_type", "table")
#             title = instructions.get("title", "Query Results")
            
#             # Execute operations safely
#             result_df = None
#             local_vars = {'self': self}
            
#             for op in operations:
#                 # Use exec instead of eval for statements
#                 if "=" in op or "print" in op:
#                     exec(op, {'pd': pd}, local_vars)
#                 else:
#                     result_df = eval(op, {'pd': pd}, local_vars)
            
#             # If no result was produced, use the final operation
#             if result_df is None:
#                 result_df = eval(operations[-1], {'pd': pd}, local_vars)
            
#             # Ensure we have a dataframe
#             if not isinstance(result_df, pd.DataFrame):
#                 if isinstance(result_df, pd.Series):
#                     result_df = result_df.to_frame()
#                 else:
#                     result_df = pd.DataFrame({'Result': [result_df]})
            
#             if output_type == "table":
#                 return {
#                     "type": "table",
#                     "data": result_df,
#                     "title": title
#                 }
#             elif output_type == "graph":
#                 graph_type = instructions.get("graph_type", "bar")
#                 fig = self.create_visualization(result_df, graph_type, title)
#                 return {
#                     "type": "graph",
#                     "figure": fig,
#                     "title": title
#                 }
#         except json.JSONDecodeError as e:
#             return {
#                 "type": "error",
#                 "message": f"Invalid JSON response: {str(e)}\nResponse: {query_instructions[:100]}..."
#             }
#         except Exception as e:
#             # Get the full traceback for debugging
#             stack_trace = traceback.format_exc()
#             return {
#                 "type": "error",
#                 "message": f"Error processing query: {str(e)}\n\nTraceback: {stack_trace}"
#             }
    
#     def create_visualization(self, df, graph_type, title):
#         """Create visualization based on the type specified"""
#         try:
#             if graph_type == "bar":
#                 if len(df.columns) >= 2:
#                     fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
#                 else:
#                     fig = px.bar(df, title=title)
#             elif graph_type == "line":
#                 if len(df.columns) >= 2:
#                     fig = px.line(df, x=df.columns[0], y=df.columns[1], title=title)
#                 else:
#                     fig = px.line(df, title=title)
#             elif graph_type == "pie":
#                 if len(df.columns) >= 2:
#                     fig = px.pie(df, names=df.columns[0], values=df.columns[1], title=title)
#                 else:
#                     fig = px.pie(df, title=title)
#             elif graph_type == "scatter":
#                 if len(df.columns) >= 3:
#                     fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=df.columns[2], title=title)
#                 elif len(df.columns) >= 2:
#                     fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title=title)
#                 else:
#                     fig = px.scatter(df, title=title)
#             else:
#                 # Default to bar chart
#                 fig = px.bar(df, title=title)
                
#             return fig
#         except Exception as e:
#             # Create a simple figure with an error message
#             fig = go.Figure()
#             fig.add_annotation(
#                 text=f"Error creating visualization: {str(e)}",
#                 xref="paper", yref="paper",
#                 x=0.5, y=0.5, showarrow=False
#             )
#             return fig
    
#     def generate_excel(self, df):
#         """Generate Excel file from dataframe"""
#         output = BytesIO()
#         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#             df.to_excel(writer, index=False, sheet_name='Results')
#         output.seek(0)
#         return output
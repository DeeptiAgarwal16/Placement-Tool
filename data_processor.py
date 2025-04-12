import pandas as pd
import json
import numpy as np
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
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess the dataframe to make it more query-friendly"""
        # Convert package/salary columns to numeric if they exist
        package_columns = [col for col in self.df.columns if any(term in col.lower() for term in ['package', 'salary', 'lpa', 'ctc'])]
        for col in package_columns:
            try:
                # Extract numeric values from strings like "10 LPA" or "₹10,00,000"
                self.df[col] = self.df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
            except Exception:
                # If conversion fails, leave as is
                pass
                
        # Convert percentage columns to numeric
        percentage_columns = [col for col in self.df.columns if any(term in col.lower() for term in ['percentage', 'cgpa', '10th', '12th', 'b.tech'])]
        for col in percentage_columns:
            try:
                self.df[col] = self.df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
            except Exception:
                # If conversion fails, leave as is
                pass
        
    def get_dataframe_info(self):
        """Get information about the dataframe structure"""
        # Check if any columns might be package/salary related
        package_columns = [col for col in self.df.columns if any(term in col.lower() for term in ['package', 'salary', 'lpa', 'ctc'])]
        
        # Check if any columns might be percentage or CGPA related
        percentage_columns = [col for col in self.df.columns if any(term in col.lower() for term in ['percentage', 'cgpa', '10th', '12th', 'b.tech'])]
        
        # Get basic statistics for numeric columns
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        stats = {}
        for col in numeric_columns:
            stats[col] = {
                "min": float(self.df[col].min()) if not pd.isna(self.df[col].min()) else 0,
                "max": float(self.df[col].max()) if not pd.isna(self.df[col].max()) else 0,
                "mean": float(self.df[col].mean()) if not pd.isna(self.df[col].mean()) else 0
            }
        
        info = {
            "columns": list(self.df.columns),
            "dtypes": {col: str(self.df[col].dtype) for col in self.df.columns},
            "sample": self.df.head(3).to_dict(orient="records"),
            "placement_status_values": list(self.df["Placement Status"].unique()) if "Placement Status" in self.df.columns else [],
            "section_values": list(self.df["Section"].unique()) if "Section" in self.df.columns else [],
            "package_columns": package_columns,
            "percentage_columns": percentage_columns,
            "statistics": stats,
            "row_count": len(self.df)
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
            local_namespace = {'self': self, 'pd': pd, 'np': np, 'df': self.df}
            
            # Execute all operations in a single context
            combined_code = "\n".join(operations)
            
            # Add a line to capture the final result
            if not any(line.strip().startswith("result_df =") for line in operations):
                combined_code += "\nresult_df = " + operations[-1]
                
            # Execute the combined code in the local namespace
            exec(combined_code, {'pd': pd, 'np': np}, local_namespace)
            
            # Try to get the result from different possible variable names
            if 'result_df' in local_namespace:
                result_df = local_namespace['result_df']
            elif 'result' in local_namespace:
                result_df = local_namespace['result']
            else:
                # Fallback - try the last operation directly
                try:
                    result_df = eval(operations[-1], {'pd': pd, 'np': np}, local_namespace)
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
                fig = self.create_visualization(result_df, graph_type, title, instructions)
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
    
    def create_visualization(self, df, graph_type, title, instructions=None):
        """Create visualization based on the type specified"""
        try:
            # Handle dictionary input for graph directly
            # if isinstance(df, dict):
            #     if graph_type == "pie":
            #         labels = list(df.keys())
            #         values = list(df.values())
            #         fig = px.pie(names=labels, values=values, title=title)
            #         return fig
            #     else:
            #         df = pd.DataFrame(list(df.items()), columns=['Category', 'Value'])
            
            # Enhanced visualization options
            if graph_type == "bar":
                if len(df.columns) >= 2:
                    x_col = instructions.get("x_column", df.columns[0]) if instructions else df.columns[0]
                    y_col = instructions.get("y_column", df.columns[1]) if instructions else df.columns[1]
                    color_col = instructions.get("color_column", None) if instructions else None
                    
                    if color_col and color_col in df.columns:
                        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title, 
                                     barmode=instructions.get("barmode", "group") if instructions else "group")
                    else:
                        fig = px.bar(df, x=x_col, y=y_col, title=title)
                else:
                    fig = px.bar(df, title=title)
                    
            elif graph_type == "line":
                if len(df.columns) >= 2:
                    x_col = instructions.get("x_column", df.columns[0]) if instructions else df.columns[0]
                    y_col = instructions.get("y_column", df.columns[1]) if instructions else df.columns[1]
                    color_col = instructions.get("color_column", None) if instructions else None
                    
                    if color_col and color_col in df.columns:
                        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
                    else:
                        fig = px.line(df, x=x_col, y=y_col, title=title)
                else:
                    fig = px.line(df, title=title)
                    
            elif graph_type == "pie":
                if len(df.columns) >= 2:
                    names_col = instructions.get("names_column", df.columns[0]) if instructions else df.columns[0]
                    values_col = instructions.get("values_column", df.columns[1]) if instructions else df.columns[1]
                    fig = px.pie(df, names=names_col, values=values_col, title=title)
                else:
                    fig = px.pie(df, title=title)
                    
            elif graph_type == "scatter":
                if len(df.columns) >= 3:
                    x_col = instructions.get("x_column", df.columns[0]) if instructions else df.columns[0]
                    y_col = instructions.get("y_column", df.columns[1]) if instructions else df.columns[1]
                    color_col = instructions.get("color_column", df.columns[2]) if instructions else df.columns[2]
                    size_col = instructions.get("size_column", None) if instructions else None
                    
                    if size_col and size_col in df.columns:
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, title=title)
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
                elif len(df.columns) >= 2:
                    x_col = instructions.get("x_column", df.columns[0]) if instructions else df.columns[0]
                    y_col = instructions.get("y_column", df.columns[1]) if instructions else df.columns[1]
                    fig = px.scatter(df, x=x_col, y=y_col, title=title)
                else:
                    fig = px.scatter(df, title=title)
                    
            elif graph_type == "heatmap":
                # For correlation heatmaps
                if "corr" in df.columns.str.lower().tolist() or df.shape[0] == df.shape[1]:
                    # This is likely a correlation matrix
                    fig = px.imshow(df, text_auto=True, aspect="auto", title=title, color_continuous_scale="Blues")
                else:
                    # Try to pivot the data for a heatmap
                    try:
                        x_col = instructions.get("x_column", df.columns[0]) if instructions else df.columns[0]
                        y_col = instructions.get("y_column", df.columns[1]) if instructions else df.columns[1]
                        z_col = instructions.get("z_column", df.columns[2]) if instructions and len(df.columns) > 2 else df.columns[2] if len(df.columns) > 2 else None
                        
                        if z_col:
                            pivot_df = df.pivot(index=y_col, columns=x_col, values=z_col)
                            fig = px.imshow(pivot_df, text_auto=True, aspect="auto", title=title, color_continuous_scale="Blues")
                        else:
                            fig = px.imshow(df, text_auto=True, aspect="auto", title=title, color_continuous_scale="Blues")
                    except Exception:
                        # Fallback to showing the data as is
                        fig = px.imshow(df, text_auto=True, aspect="auto", title=title, color_continuous_scale="Blues")
                        
            elif graph_type == "box":
                if len(df.columns) >= 2:
                    x_col = instructions.get("x_column", df.columns[0]) if instructions else df.columns[0]
                    y_col = instructions.get("y_column", df.columns[1]) if instructions else df.columns[1]
                    color_col = instructions.get("color_column", None) if instructions else None
                    
                    if color_col and color_col in df.columns:
                        fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title)
                    else:
                        fig = px.box(df, x=x_col, y=y_col, title=title)
                else:
                    fig = px.box(df, title=title)
                    
            else:
                # Default to bar chart
                fig = px.bar(df, title=title)
                
            return fig
        except Exception as e:
            # Create a simple figure with an error message
            fig = go.Figure()
            fig = px.bar(df, x=df.iloc[:, 0], y=df.iloc[:, 1], title=title)
            # fig.add_annotation(
            #     text=f"Error creating visualization: {str(e)}",
            #     xref="paper", yref="paper",
            #     x=0.5, y=0.5, showarrow=False
            # )
            return fig


    
    def generate_excel(self, df):
        """Generate Excel file from dataframe"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        output.seek(0)
        return output
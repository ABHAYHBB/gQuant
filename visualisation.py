import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_strip_plot(df1, columns_to_plot):
    """
    This function creates a strip plot with additional details from a given DataFrame.
    
    Parameters:
    df1 (pd.DataFrame): The DataFrame containing the data.
    columns_to_plot (list): List of column names from the DataFrame to be included in the strip plot.
    
    Returns:
    None: The function displays the plot.
    """
    # Initialize the figure
    fig = go.Figure()

    # Add the strip plots
    for i, col in enumerate(columns_to_plot):
        fig.add_trace(go.Box(y=df1[col], name=col, jitter=0.3, pointpos=-1.8, boxpoints='all'))

    # Add mean lines
    for i, col in enumerate(columns_to_plot):
        mean_val = df1[col].mean()
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=-0.5 + i,
                x1=0.5 + i,
                y0=mean_val,
                y1=mean_val,
                yref="y",
                xref="x",
                line=dict(color="Red", dash="dashdot")
            )
        )

    # Additional settings, including figure size
    fig.update_layout(
        title="Strip Plot with Additional Details",
        xaxis_title="Columns",
        yaxis_title="Value",
        width=1300,  # Width in pixels
        height=600   # Height in pixels
    )

    # Show the plot
    fig.show()














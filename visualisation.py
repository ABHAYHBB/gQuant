import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import norm
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import LabelSet, ColumnDataSource

def create_kde_plot(df1, bandwidth=1.0):
    """
    This function creates a KDE (Kernel Density Estimation) plot from a given DataFrame.
    
    Parameters:
    df1 (pd.DataFrame): The DataFrame containing the data.
    bandwidth (float): The bandwidth for the KDE calculation.
    
    Returns:
    None: The function displays the plot.
    """
    # KDE Function
    def kernel_density_estimation(x, data_points, bandwidth):
        n = len(data_points)
        result = 0
        for xi in data_points:
            u = (x - xi) / bandwidth
            result += norm.pdf(u)  # Gaussian kernel
        return result / (n * bandwidth)
    
    # Create a Bokeh figure
    p = figure(title='Kernel Density Estimation (KDE) Plot', x_axis_label='Value', y_axis_label='Density',
               width=900, height=750, output_backend="webgl")

    # Container for label data
    label_data = []

    # Generate KDE values for each column and add to the Bokeh plot
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # Define colors for the lines
    for i, column in enumerate(df1.columns):
        x_values = np.linspace(df1[column].min(), df1[column].max(), 100)
        y_values = [kernel_density_estimation(x, df1[column], bandwidth) for x in x_values]
        p.line(x_values, y_values, legend_label=column, line_width=3, line_color=colors[i % len(colors)])

        # Choose a point to label (e.g., the peak of the KDE curve)
        idx_max = np.argmax(y_values)
        x_max = x_values[idx_max]
        y_max = y_values[idx_max]

        # Store label data (only KDE values this time)
        label_data.append({'x': x_max, 'y': y_max, 'text': f"{y_max:.2f}"})

    # Separate the label data into distinct lists for each 'column'
    x_vals, y_vals, texts = [], [], []
    for label in label_data:
        x_vals.append(label['x'])
        y_vals.append(label['y'])
        texts.append(label['text'])

    # Create a ColumnDataSource from the label data
    source = ColumnDataSource(data={'x': x_vals, 'y': y_vals, 'text': texts})

    # Add LabelSet
    labels = LabelSet(x='x', y='y', text='text', level='glyph', source=source)
    p.add_layout(labels)

    # Output to notebook
    output_notebook()

    # Show the plot
    show(p)

def create_box_plot(df1):
    """
    This function creates a box plot with jittered points from a given DataFrame.
    
    Parameters:
    df1 (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
    None: The function displays the plot.
    """
    # Initialize the figure
    fig = go.Figure()

    # Add the strip plots with new names
    for column in df1.columns:
        fig.add_trace(go.Box(y=df1[column], name=column, jitter=0.3, pointpos=-1.8, boxpoints='all'))

    # Add mean lines with new names
    for i, column in enumerate(df1.columns):
        mean_val = df1[column].mean()
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
        title="Box Plot",
        xaxis_title="Features",
        yaxis_title="Values",
        width=1000,
        height=900
    )

    # Show the plot
    fig.show()

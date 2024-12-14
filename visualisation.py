import pandas as pd
import numpy as np
from scipy.stats import norm
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import LabelSet, ColumnDataSource

# KDE Function
def kernel_density_estimation(x, data_points, bandwidth):
    n = len(data_points)
    result = 0
    for xi in data_points:
        u = (x - xi) / bandwidth
        result += norm.pdf(u)  # Gaussian kernel
    return result / (n * bandwidth)

# Read the Excel data into a DataFrame
df = pd.read_excel('endo_2.xlsx',sheet_name='Sheet2')

# Specify the column names to plot
columns_to_plot = ['Ct Mean (Let-7c-5p)', '16-5p', '30-a', '30-d', '10b-5p']

# Define colors for the lines
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Bandwidth
bandwidth = 1.0

# Create a Bokeh figure
p = figure(title='Kernel Density Estimation (KDE) Plot', x_axis_label='Value', y_axis_label='Density',
           width=900, height=750, output_backend="webgl")


# Container for label data
label_data = []

# Generate KDE values for each column and add to the Bokeh plot
for i, column in enumerate(columns_to_plot):
    x_values = np.linspace(df[column].min(), df[column].max(), 100)
    y_values = [kernel_density_estimation(x, df[column], bandwidth) for x in x_values]
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

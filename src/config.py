import os
import plotly.express as px

# path to the folder containing the raw data CSV
data_dir = 'data'

# full path to the telecom churn dataset
DATA_PATH = os.path.join(data_dir, 'telco-customer-churn.csv')

# folder where all output files (plots, artifacts, scores) will be saved
OUTPUT_DIR = 'outputs/'

# configuration for all plots in the project
PLOT_CONFIG = {
    # settings for pie and donut charts
    'pie': {
        'hole': 0.4, # size of center hole (0 to 1)
        'textinfo': 'percent+label', # show percentage and label
        'font_size': 14, # size of text inside pie
        'width': 800, # plot width in pixels
        'height': 400, # plot height in pixels
        'color_sequence': px.colors.qualitative.Pastel # list of pastel colors
    },

    # settings for histograms and bar charts
    'hist': {
        'width': 700, # plot width in pixels
        'height': 500, # plot height in pixels
        'bargap': 0.15, # gap between bars (0 to 1)
        'color_sequence': px.colors.qualitative.Set3 # list of soft pastel colors
    },

    # color pairs for kernel density plots by churn status
    'kde_colors': {
        'MonthlyCharges': ('#AEC6CF', '#FFB347'), # pastel blue, pastel orange
        'TotalCharges': ('#77DD77', '#FFD1DC') # pastel green, pastel pink
    },

    # manual color mapping for binary features and gender
    'bar_colors': {
        'binary': {
            'Yes': '#FFB3BA', # pastel pink for "Yes"
            'No': '#B5EAEA' # pastel blue for "No"
        },
        'gender': {
            'Male': '#C1E1C1', # pastel mint for "Male"
            'Female': '#F5C6EA' # pastel lavender for "Female"
        }
    }
}

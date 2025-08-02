import os
import pandas as pd

def load_data(path):
    """
    Read a CSV file into a pandas DataFrame.
    - path: file path to the CSV dataset
    - returns: DataFrame containing the CSV data
    """
    return pd.read_csv(path)


def save_fig(fig, filename, **kwargs):
    """
    Save a Matplotlib or Plotly figure to disk.
    - fig: a Matplotlib figure or Plotly Figure object
    - filename: full path where the image will be saved
    - kwargs: additional save parameters (dpi, bbox_inches for Matplotlib)
    
    This function will:
    - create the output directory if it does not exist
    - use fig.write_image for Plotly figures
    - fall back to fig.savefig for Matplotlib figures
    """
    # ensure output folder exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # if Plotly figure has write_image, use it
    if hasattr(fig, 'write_image'):
        fig.write_image(filename, **kwargs)
    else:
        # otherwise save with Matplotlib's savefig
        fig.savefig(
            filename,
            dpi=kwargs.get('dpi', 100),
            bbox_inches=kwargs.get('bbox_inches', 'tight')
        )

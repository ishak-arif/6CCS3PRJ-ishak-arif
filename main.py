import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore', category=FutureWarning)

from pipeline import run_pipeline
from results import export_results
from plots import generate_plots

if __name__ == '__main__':
    r = run_pipeline()
    export_results(r)
    generate_plots(r)
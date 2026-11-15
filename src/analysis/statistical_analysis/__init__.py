"""
Statistical analysis module for sentiment predictions.

Supports multiple dependent variables and multiple model batches.

Usage:
    python -m src.analysis.statistical_analysis.main \
        --batch-paths path1:model1,path2:model2 \
        --leaders-path data/LeadersAndParties/leaders.csv \
        --dependent-vars v2xpa_popul_bool,v2paclient_bool
"""

from .main import run_analysis
from .data_loader import load_and_prepare_data
from .stat_models import fit_ordinal_model, fit_ols_model, fit_all_models, ModelResults, AllModelResults
from .analysis import compute_all_summary_statistics, print_all_summary_statistics
from .visualisation.visualization import plot_common_figures, plot_all_stat_figures
from .config import BOOL_INDEPENDENT_VARS

__all__ = [
    "run_analysis",
    "load_and_prepare_data",
    "fit_ordinal_model",
    "fit_ols_model",
    "fit_all_models",
    "ModelResults",
    "AllModelResults",
    "compute_all_summary_statistics",
    "print_all_summary_statistics",
    "BOOL_INDEPENDENT_VARS",
    "plot_common_figures",
    "plot_all_stat_figures",
]

"""Compact FAccT Figure 2; reuse the paper's data preparation without fitting models.

Run from encyclopedia_analysis:
    MPLCONFIGDIR=/tmp/facct-mpl python -m src.analysis.statistical_analysis.facct_score_distribution
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from .config import CONTINUOUS_INDEPENDENT_VARS, MODEL_DISPLAY, SENTIMENT_COLORS
from .data_loader import load_and_prepare_data


def main():
    project = Path(__file__).resolve().parents[3]
    analysis = project
    output = project / 'results/figures/descriptive_stats'
    df = load_and_prepare_data(
        analysis / 'results/final2.0',
        analysis / 'data/articles/LeadersAndParties/leaders_processed.csv',
        'ideology', continuous_independent_vars=CONTINUOUS_INDEPENDENT_VARS,
    )
    source = pd.crosstab(df.Source, df.neutrality_assessment, normalize='index') * 100
    model = pd.crosstab(df.Model, df.neutrality_assessment, normalize='index') * 100
    shares = pd.concat([
        source.reindex(['grokipedia_0.2', 'wikipedia']),
        model.reindex(list(MODEL_DISPLAY)),
    ]).reindex(columns=[-2, -1, 0, 1, 2], fill_value=0)
    # Guard against quietly rendering a newer/different sample than the paper.
    np.testing.assert_allclose(shares.sum(axis=1), 100)
    np.testing.assert_array_equal(shares[0].round(1), [61.0, 70.3, 25.0, 67.8, 82.9, 86.9])
    labels = ['Grokipedia', 'Wikipedia', *MODEL_DISPLAY.values()]
    y = np.array([0, 1, 2.5, 3.5, 4.5, 5.5])
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(6, 1.75))
        fig.subplots_adjust(left=.18, right=.975, top=.97, bottom=.29)
        left = np.zeros(6)
        for score, color in zip(shares.columns, SENTIMENT_COLORS):
            values = shares[score].to_numpy()
            ax.barh(y, values, left=left, height=.72, color=color,
                    edgecolor='white', linewidth=.35)
            left += values
        # Label the four largest segments in-place, omitting any below 3%.
        # The smallest tails stay unlabelled rather than crowding the chart.
        for row, values in zip(y, shares.to_numpy()):
            centers = np.cumsum(values) - values / 2
            for index in np.argsort(values)[-4:]:
                value = values[index]
                if value < 3:
                    continue
                fontsize = 6 if value < 4 else (7 if value < 8 else 8)
                ax.text(centers[index], row, f'{value:.1f}',
                        ha='center', va='center', fontsize=fontsize)
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_ylim(6.05, -.6)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis='both', length=0, pad=3, labelsize=7)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel('Share of ratings (%)', fontsize=8, labelpad=2)
        ax.axhline(1.75, color='.8', linewidth=.5)
        for tag, row in [('(a)', .5), ('(b)', 4)]:
            ax.text(-.205, row, tag, transform=ax.get_yaxis_transform(),
                    ha='left', va='center', fontsize=8, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_visible(False)
        legends = ['−2 strongly against', '−1', '0 neutral', '+1', '+2 strongly in favor']
        fig.legend([Patch(facecolor=c) for c in SENTIMENT_COLORS], legends,
                   loc='lower center', bbox_to_anchor=(.5, -.005), ncol=5,
                   fontsize=7, frameon=False, handlelength=1.1,
                   handletextpad=.4, columnspacing=1.1)
        output.mkdir(parents=True, exist_ok=True)
        for ext in ['pdf', 'png']:
            fig.savefig(output / f'00_score_distribution_facct.{ext}', dpi=220)
        shares.to_csv(output / '00_score_distribution_facct.csv', index_label='group')
        plt.close(fig)
    print(shares.round(1).to_string())


if __name__ == '__main__':
    main()

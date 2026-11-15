import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from typing import Optional

from ..config import PLOT_STYLE, MODEL_DISPLAY, SOURCE_DISPLAY

SOURCE_COLUMN = "Source"
MODEL_COLUMN = "Model"
OUTCOME_COLUMN = "neutrality_assessment"


def pretty_model(value: str) -> str:
    return MODEL_DISPLAY.get(str(value), str(value))


def pretty_source(value: str) -> str:
    return SOURCE_DISPLAY.get(str(value), str(value))


def _maybe_map(text: str) -> str:
    return MODEL_DISPLAY.get(text, SOURCE_DISPLAY.get(text, text))


def relabel_model_source(ax: plt.Axes) -> None:
    """Rewrite tick labels and legend entries using MODEL_DISPLAY / SOURCE_DISPLAY."""
    for axis in (ax.xaxis, ax.yaxis):
        labels = [t.get_text() for t in axis.get_ticklabels()]
        if any(label in MODEL_DISPLAY or label in SOURCE_DISPLAY for label in labels):
            axis.set_ticklabels([_maybe_map(label) for label in labels])

    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_text(_maybe_map(text.get_text()))


def save_figure(
    fig: plt.Figure,
    filepath: Path,
    dpi: int = 150,
    bbox_inches='tight',
    pad_inches: Optional[float] = None,
    transparent: bool = False,
) -> None:
    """Write *fig* to *filepath*, creating parent directories as needed.

    ``pad_inches`` overrides matplotlib's default 0.1 in border added around a
    ``bbox_inches="tight"`` crop. Pass a near-zero value for figures authored at
    an exact printed width (e.g. a paper's \\columnwidth), where 0.2 in of extra
    whitespace would force LaTeX to scale the figure — and its fonts — down.

    ``transparent=True`` drops the figure's and the axes' background patches, so
    the file carries only the marks. On a slide that lets the deck's own
    background show through instead of a white card sitting on it; in the paper
    it makes no visible difference against a white page.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {} if pad_inches is None else {"pad_inches": pad_inches}
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches,
                transparent=transparent, **kwargs)
    print(f"Saved: {filepath}")


def setup_plot_style() -> None:
    sns.set_style(PLOT_STYLE)


def compute_article_length(text: Optional[str]) -> Optional[int]:
    if not isinstance(text, str):
        return None
    return len(text.split())


_DEP_LABEL_RENAMES = {
    "culsup": "culincl",
    "relig": "secular",
    "anticlient": "anti-client",
    "antipopul": "anti-popul",
    "riglef": "rightwing",
    "is_Grokipedia": "Grokipedia",
    "article_length_diff": "Δ article length",
    "article_length": "Article length",
}


def pretty_dep_label(name: str) -> str:
    """Return a short, human-friendly version of a dependent variable name.

    Translates ``Model_<raw>`` (and ``C(Model)[T.<raw>]`` formula form) dummy
    column names to ``Model_<DisplayName>`` using ``MODEL_DISPLAY``, so labels
    generalise to any rater rather than hardcoding a single one.
    """
    stripped = (
        name.replace("_bool", "")
        .replace("v2xpa_", "")
        .replace("v2pa", "")
        .replace("_osp", "")
    )
    for raw, display in MODEL_DISPLAY.items():
        stripped = stripped.replace(f"Model_{raw}", f"Model_{display}")
        stripped = stripped.replace(f"C(Model)[T.{raw}]", f"Model_{display}")
    for old, new in _DEP_LABEL_RENAMES.items():
        stripped = stripped.replace(old, new)
    # Display interaction terms with "*" instead of the internal "_x_" token.
    stripped = stripped.replace("_x_", "*")
    return stripped


# Long-form ideology names for a deck, keyed by the short label
# `pretty_dep_label` produces. Slides have the width the paper does not, and an
# audience reading a figure once cannot expand "culincl" the way a reader with
# the variable table in front of them can. Display-only: the fits, the ordering
# and the term-type styling all still key off the short labels.
SLIDE_IDEOLOGY_LABELS = {
    "culincl": "Cultural Inclusivity",
    "immig": "Immigration Openness",
    "lgbt": "LGBT Equality",
    "womlab": "Women's Labour Equality",
    "anti-client": "Anti-Clientelism",
    "plural": "Pluralism",
    "anti-popul": "Anti-Populism",
    "secular": "Secular Positioning",
    "rightwing": "Right-Wing",
}

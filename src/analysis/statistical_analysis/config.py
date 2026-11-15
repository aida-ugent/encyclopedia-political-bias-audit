"""Configuration constants for statistical analysis."""

from pathlib import Path

LIKERT_MAPPING = {
    "strongly_biased_against": -2,
    "somewhat_biased_against": -1,
    "fully_neutral": 0,
    "somewhat_biased_in_favor": 1,
    "strongly_biased_in_favor": 2,
}

SOURCE_NAME_MAPPING = {
    "wikipedia": "wikipedia",
    "grokipedia_0_2": "grokipedia_0.2",
    "grokipedia_0_1": "grokipedia_0.1",
}

DEFAULT_LEADERS_PATH = "data/articles/LeadersAndParties/leaders_processed.csv"
BOOL_INDEPENDENT_VARS = ["v2paanticlient_osp_bool", "v2paculsup_osp_bool", "v2paimmig_osp_bool", "v2palgbt_osp_bool", "v2pawomlab_osp_bool", "v2xpa_plural_bool", "v2xpa_antipopul_bool",  "v2parelig_osp_bool", "v2pariglef_osp_bool"]  #
CONTINUOUS_INDEPENDENT_VARS = ["v2paanticlient_osp", "v2paculsup_osp", "v2paimmig_osp", "v2palgbt_osp", "v2pawomlab_osp", "v2xpa_plural", "v2xpa_antipopul", "v2parelig_osp", "v2pariglef_osp"] # ""
# BOOL_INDEPENDENT_VARS = ["v2xpa_antiplural_bool"]
# CONTINUOUS_INDEPENDENT_VARS = [ "v2paimmig_osp", "v2pawomlab_osp", "v2palgbt_osp"]
# --------------------------------------------------------------------------
# Scale geometry of the ideology dimensions (V-Party codebook v2, Feb 2022)
# --------------------------------------------------------------------------
# Needed only by the magnitude models, where the outcome is |Y| (distance from
# neutral) rather than the signed rating. A direction-free outcome has to be
# paired with a direction-free covariate, but "direction-free" is only
# meaningful for dimensions whose response scale actually has two opposed
# poles with a neutral category between them. The V-Party codebook splits the
# nine dimensions cleanly in two:
#
#   BIPOLAR  - the scale runs from one position, through an explicit
#              no-position / centre category, to the opposing position. The
#              codebook midpoint is a substantive "takes no side", so
#              ideological commitment is U-shaped in the raw score and the
#              magnitude-relevant covariate is |x - c| (extremity).
#
#   MONOTONE - the scale is a frequency or extent ladder ("not at all" ->
#              "always"). There is no opposing pole: a low score is the
#              absence of the behaviour, not its mirror image. Extremity is
#              not defined, so these keep their signed level.
#
# RAW_SUFFIX marks the untouched original-scale copies that
# ``data_loader.process_leaders_df`` keeps alongside the z-scored columns; the
# midpoints below are on that original scale, not on the z-scale.
RAW_SUFFIX = "__raw"

# var -> codebook neutral point on the original (_osp) scale.
#   v2paculsup  0 strongly promotes .. 2 "Ambiguous"          .. 4 strongly opposes
#   v2paimmig   0 strongly opposes  .. 2 "Ambiguous/No pos."  .. 4 strongly supports
#   v2palgbt    0 strongly opposes  .. 2 "Ambiguous/No pos."  .. 4 strongly supports
#   v2pawomlab  0 strongly opposes  .. 2 "Ambiguous/No pos."  .. 4 strongly supports
#   v2pariglef  0 far-left          .. 3 "Center"             .. 6 far-right
BIPOLAR_NEUTRAL_POINT = {
    "v2paculsup_osp": 2.0,
    "v2paimmig_osp": 2.0,
    "v2palgbt_osp": 2.0,
    "v2pawomlab_osp": 2.0,
    "v2pariglef_osp": 3.0,
}

# Monotone dimensions, with the codebook item behind each one:
#   v2paanticlient_osp  reversed v2paclient: 0 "not at all" .. 4 "as its main effort"
#   v2parelig_osp       0 "always invokes religion" .. 4 "never"
#   v2xpa_plural        1 - v2xpa_antiplural, built from four 0-4 commitment items
#   v2xpa_antipopul     1 - v2xpa_popul, built from two 0-4 frequency/importance items
MONOTONE_DIMENSIONS = [
    "v2paanticlient_osp",
    "v2parelig_osp",
    "v2xpa_plural",
    "v2xpa_antipopul",
]

PLOT_STYLE = "whitegrid"
# Diverging 5-step ramp for the ordinal neutrality score, in outcome order
# (-2, -1, 0, +1, +2). The poles are the two paper-wide sign colors — gold for
# "biased against", teal for "biased in favor" — matching _NEGATIVE_COLOR /
# _POSITIVE_COLOR in visualisation/stat_figures.py; the +-1 steps are lighter
# tints of the same hues, and 0 is a light neutral so the (dominant) neutral
# band recedes and the biased slices carry the ink.
SENTIMENT_COLORS = ["#ebc838", "#f6e6a8", "#cccccc", "#7ec8d9", "#0097b2"]

# Insertion order doubles as the canonical model ordering for model-grouped
# figures (e.g. 00_score_distribution_by_model): Claude, Grok, Mistral, DeepSeek.
# Keep this order: it happens to be strictly increasing in neutral rate
# (25.0 -> 67.8 -> 82.9 -> 86.9 on the final2.0 run), so the bars in panel (b)
# of 00_score_distribution_by_source_and_model form a monotone gradient rather
# than an arbitrary sequence. Reordering costs that for free.
MODEL_DISPLAY = {
    "claude-opus-4-6": "Claude",
    "grok-4": "Grok",
    "mistralai": "Mistral",
    "deepseek-v4-pro": "DeepSeek",
}
# Reference (baseline) model for all Model-factor encodings — both the OLS
# C(Model) treatment coding and the ordinal get_dummies(drop_first) use the
# first category as the dropped reference, so this is enforced via category order.
BASELINE_MODEL = "grok-4"

# Reference model for the per-source OLS regressions only (overrides the
# global BASELINE_MODEL category order within that loop).
PER_SOURCE_BASELINE_MODEL = "mistralai"

# Plain main-effect control covariates (no interactions). Added to the
# differential regressions and grouped/styled as controls on the coefplots.
CONTROL_VARS = ["article_length_diff"]

# Regional fixed effects: collapse the country_isocode levels into world regions
# used as dummy controls (e.g. in the per-source OLS grid).
#
# Grouping is V-Dem's politico-geographic 6-region scheme (e_regionpol_6C),
# derived directly from data/LeadersAndParties/V-Dem-CPD-Party-V2.csv so it stays
# consistent with the V-Dem ideology variables used elsewhere. Note the V-Dem
# conventions: Cyprus, Australia and New Zealand sit in "the West"; Turkey,
# Israel in MENA; Mongolia in Eastern Europe & Central Asia; Sudan in SSA.
REGION_COLUMN = "region"
# Dropped reference level: "the West" (Western Europe, North America, Oceania) —
# the conventional comparative-politics baseline and the best-covered group for
# English-language articles.
REGION_BASELINE = "Western Europe, North America and Oceania"
REGIONS = {
    "Sub-Saharan Africa": [
        "AGO", "BDI", "BEN", "BFA", "BWA", "CAF", "CIV", "CMR", "COD",
        "COG", "COM", "CPV", "DJI", "ERI", "ETH", "GAB", "GHA", "GIN",
        "GMB", "GNB", "GNQ", "KEN", "LBR", "LSO", "MDG", "MLI", "MOZ",
        "MRT", "MUS", "MWI", "NAM", "NER", "NGA", "RWA", "SDN", "SEN",
        "SLE", "SML", "SOM", "STP", "SWZ", "SYC", "TCD", "TGO", "TZA",
        "UGA", "ZAF", "ZMB", "ZWE", "ZZB",
    ],
    "Eastern Europe and Central Asia": [
        "ALB", "ARM", "AZE", "BGR", "BIH", "BLR", "CZE", "DDR", "EST",
        "GEO", "HRV", "HUN", "KAZ", "KGZ", "LTU", "LVA", "MDA", "MKD",
        "MNE", "MNG", "POL", "ROU", "RUS", "SRB", "SVK", "SVN", "TJK",
        "TKM", "UKR", "UZB", "XKX",
    ],
    "Latin America and the Caribbean": [
        "ARG", "BOL", "BRA", "BRB", "CHL", "COL", "CRI", "CUB", "DOM",
        "ECU", "GTM", "GUY", "HND", "HTI", "JAM", "MEX", "NIC", "PAN",
        "PER", "PRY", "SLV", "SUR", "TTO", "URY", "VEN",
    ],
    "Western Europe, North America and Oceania": [
        "AUS", "AUT", "BEL", "CAN", "CHE", "CYP", "DEU", "DNK", "ESP",
        "FIN", "FRA", "GBR", "GRC", "IRL", "ISL", "ITA", "LUX", "MLT",
        "NLD", "NOR", "NZL", "PRT", "SWE", "USA",
    ],
    "Asia and the Pacific": [
        "AFG", "BGD", "BTN", "CHN", "FJI", "HKG", "IDN", "IND", "JPN",
        "KHM", "KOR", "LAO", "LKA", "MDV", "MMR", "MYS", "NPL", "PAK",
        "PHL", "PNG", "PRK", "SGP", "SLB", "THA", "TLS", "TWN", "VDR",
        "VNM", "VUT",
    ],
    "The Middle East and North Africa": [
        "ARE", "BHR", "DZA", "EGY", "IRN", "IRQ", "ISR", "JOR", "KWT",
        "LBN", "LBY", "MAR", "OMN", "PSE", "SYR", "TUN", "TUR", "YEM",
        "YMD",
    ],
}
# Flattened ISO3 -> region lookup.
REGIONS_BY_ISO3 = {iso: region for region, isos in REGIONS.items() for iso in isos}
SOURCE_DISPLAY = {
    "wikipedia": "Wikipedia",
    "grokipedia_0.2": "Grokipedia",
}
SCORE_LABELS = {
    -2: "Strongly biased against (-2)",
    -1: "Biased against (-1)",
    0: "Neutral (0)",
    1: "Biased in favour (+1)",
    2: "Strongly biased in favour (+2)",
}

# Clustering unit for all cluster-robust standard errors. Politicians from the
# same party share both their V-Dem ideology scores (which are measured at the
# party level, not the individual level) and much of their editorial treatment,
# so the party — not the individual politician — is the level at which the
# residuals are plausibly independent. Built in
# ``data_loader.prepare_analysis_dataframe`` from Party Facts ids, falling back
# to country + party name where an id is missing.
CLUSTER_COLUMN = "party_id"

ORDINAL_DISTRIBUTION = "logit"
ORDINAL_METHOD = "bfgs"
OLS_COV_TYPE = "cluster"


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent


def get_figures_dir(results_dir: Path) -> Path:
    """Create and return figures subdirectory."""
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir

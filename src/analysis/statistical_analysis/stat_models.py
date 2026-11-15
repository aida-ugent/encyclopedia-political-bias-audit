"""Statistical models for sentiment analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import expit
from scipy.stats import norm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.anova import anova_lm

from .config import ORDINAL_DISTRIBUTION, ORDINAL_METHOD, OLS_COV_TYPE, CLUSTER_COLUMN


@dataclass
class ModelResults:
    """Container for model fitting results."""
    model_type: str
    results: object
    predictors: Optional[List[str]] = None
    anova_table: Optional[pd.DataFrame] = None
    independent_vars: Optional[List[str]] = None  # All dep vars included in model
    marginal_effects: Optional[object] = None  # OrdinalMargEffResults for ordinal model


@dataclass
class AllModelResults:
    """Container for single OLS and single ordinal model results."""
    ordinal: Optional[ModelResults] = None
    ols: Optional[ModelResults] = None


@dataclass
class OrdinalMargEffResults:
    """Average marginal effects for the ordinal logit model."""
    margeff: np.ndarray      # (n_pred, n_cat)
    margeff_se: np.ndarray   # (n_pred, n_cat)
    pvalues: np.ndarray      # (n_pred, n_cat)


def compute_ordinal_margeff(res, predictor_names: List[str]) -> Optional[OrdinalMargEffResults]:
    """
    Compute average marginal effects (AME) for an ordinal logit model.

    For each predictor k and outcome category j:
        AME_k(j) = (1/N) Σ_i [λ(τ_{j-1} - X_i'β) - λ(τ_j - X_i'β)] · β_k
    where λ is the logistic PDF and τ are the model thresholds.

    Standard errors are computed via the delta method with a numerical Jacobian.
    """
    try:
        X = res.model.exog
        n_obs, n_pred = X.shape
        params_vals = res.params.values
        n_thresh = len(params_vals) - n_pred
        n_cat = n_thresh + 1
        cov = res.cov_params().values

        def _logistic_pdf(z: np.ndarray) -> np.ndarray:
            p = expit(z)
            return p * (1.0 - p)

        def _ame_flat(params: np.ndarray) -> np.ndarray:
            beta = params[:n_pred]
            thresholds = res.model.transform_threshold_params(params)
            tau_ext = np.concatenate([[-np.inf], thresholds, [np.inf]])
            Xbeta = X @ beta
            ame = np.zeros((n_pred, n_cat))
            for j in range(n_cat):
                tau_lo, tau_hi = tau_ext[j], tau_ext[j + 1]
                pdf_lo = (np.zeros(n_obs) if np.isinf(tau_lo)
                          else _logistic_pdf(tau_lo - Xbeta))
                pdf_hi = (np.zeros(n_obs) if np.isinf(tau_hi)
                          else _logistic_pdf(tau_hi - Xbeta))
                scale = (pdf_lo - pdf_hi).mean()
                ame[:, j] = scale * beta
            return ame.ravel()

        ame_flat = _ame_flat(params_vals)
        ame = ame_flat.reshape(n_pred, n_cat)

        # Numerical Jacobian for delta-method SEs
        eps = 1e-5
        n_params = len(params_vals)
        jac = np.zeros((n_pred * n_cat, n_params))
        for i in range(n_params):
            p_plus = params_vals.copy(); p_plus[i] += eps
            p_minus = params_vals.copy(); p_minus[i] -= eps
            jac[:, i] = (_ame_flat(p_plus) - _ame_flat(p_minus)) / (2.0 * eps)

        var_flat = np.diag(jac @ cov @ jac.T)
        se = np.sqrt(np.maximum(var_flat, 0.0)).reshape(n_pred, n_cat)

        z_stat = np.divide(ame, se, out=np.zeros_like(ame), where=se > 0)
        pvalues = 2.0 * (1.0 - norm.cdf(np.abs(z_stat)))

        return OrdinalMargEffResults(margeff=ame, margeff_se=se, pvalues=pvalues)

    except Exception as exc:
        print(f"Warning: Could not compute marginal effects: {exc}")
        return None


def create_dummy_variables(df: pd.DataFrame, columns: List[str], drop_first: bool = True) -> pd.DataFrame:
    """Create dummy variables for categorical columns."""
    existing_cols = [c for c in columns if c in df.columns]
    return pd.get_dummies(df, columns=existing_cols, drop_first=drop_first)


def create_interaction_terms(
    df: pd.DataFrame,
    factor_groups: List[Tuple[List[str], List[str]]],
) -> Tuple[pd.DataFrame, List[str]]:
    """Create interaction terms between groups of factor columns."""
    df = df.copy()
    interaction_cols = []
    for group1, group2 in factor_groups:
        print(f"Creating interactions between groups: {group1} and {group2}")
        for col1 in group1:
            for col2 in group2:
                print(f"Processing interaction: {col1} x {col2}")
                inter_col = f"{col1}_x_{col2}"
                df[inter_col] = df[col1] * df[col2]
                interaction_cols.append(inter_col)
    return df, interaction_cols


def fit_ordinal_model(
    df: pd.DataFrame,
    independent_vars: List[str],
    outcome_col: str = "neutrality_assessment",
    cluster_col: Optional[str] = CLUSTER_COLUMN,
    model_col: str = "Model",
    source_col: str = "Source",
    use_source: bool = True,
    use_model: bool = False,
    use_model_dep_interactions: bool = True,
    include_model_term: bool = True,
    control_vars: Optional[List[str]] = None,
) -> Optional[ModelResults]:
    """
    Fit a single ordinal regression model with all dependent variables as predictors.

    `control_vars` are added as plain main-effect covariates (no interactions).
    `use_model_dep_interactions=False` keeps the judge dummies as main effects
    only (no Model x ideology terms), mirroring the OLS switch of the same name.

    Returns ModelResults or None if fitting fails.
    """
    # Filter to dep_vars that exist in df
    existing_indep_vars = [iv for iv in independent_vars if iv in df.columns]
    if not existing_indep_vars:
        return None

    # Get columns for each dep_var
    all_dep_cols = []
    for dv in existing_indep_vars:
        dep_cols = [c for c in df.columns if c == dv]
        all_dep_cols.extend(dep_cols)

    # Create two-way interactions: Model*dep_vars, Source*dep_vars, Model*Source
    if use_source:
        if include_model_term:
            # One-hot encode model_col when it is non-numeric so the ordinal
            # design matrix stays numeric and the model generalises to >2
            # raters (N-1 dummies).
            if not pd.api.types.is_numeric_dtype(df[model_col]):
                model_dummies = pd.get_dummies(
                    df[model_col], prefix=model_col, drop_first=True
                ).astype(float)
                model_dummy_cols = list(model_dummies.columns)
                df = df.copy()
                for col in model_dummy_cols:
                    df[col] = model_dummies[col].values
            else:
                model_dummy_cols = [model_col]
            # Match the OLS specification exactly: Model main effect, Source
            # main effect, dep main effects, plus Model×dep, Source×dep and
            # Model×Source interactions.
            interaction_groups = [
                ([source_col], all_dep_cols),
                (model_dummy_cols, [source_col]),
            ]
            if use_model_dep_interactions:
                interaction_groups.insert(0, (model_dummy_cols, all_dep_cols))
            df, interaction_cols = create_interaction_terms(df, interaction_groups)
            predictors = model_dummy_cols + [source_col] + all_dep_cols + interaction_cols
        else:
            interaction_groups = [
                ([source_col], all_dep_cols),
            ]
            df, interaction_cols = create_interaction_terms(df, interaction_groups)
            predictors = [source_col] + all_dep_cols + interaction_cols
    elif use_model:
        # One-hot encode model_col (categorical) then interact dummies with dep vars
        model_dummies = pd.get_dummies(df[model_col], prefix=model_col, drop_first=True).astype(float)
        dummy_cols = list(model_dummies.columns)
        df = df.copy()
        for col in dummy_cols:
            df[col] = model_dummies[col].values
        if use_model_dep_interactions:
            df, interaction_cols = create_interaction_terms(df, [(dummy_cols, all_dep_cols)])
        else:
            interaction_cols = []
        predictors = dummy_cols + all_dep_cols + interaction_cols
    else:
        predictors = all_dep_cols

    # Plain main-effect controls (no interactions). Numeric controls enter
    # directly; categorical ones are one-hot encoded (N-1 dummies).
    if control_vars:
        control_cols = []
        for cv in control_vars:
            if cv not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[cv]):
                control_cols.append(cv)
            else:
                cv_dummies = pd.get_dummies(df[cv], prefix=cv, drop_first=True).astype(float)
                df = df.copy()
                for col in cv_dummies.columns:
                    df[col] = cv_dummies[col].values
                    control_cols.append(col)
        predictors = predictors + control_cols

    X = df[predictors].astype(float)
    y = df[outcome_col].astype(int)

    print(X)
    model = OrderedModel(y, X, distr=ORDINAL_DISTRIBUTION)
    fit_kwargs = {"method": ORDINAL_METHOD}
    if cluster_col is not None and cluster_col in df.columns:
        fit_kwargs["cov_type"] = "cluster"
        fit_kwargs["cov_kwds"] = {"groups": df[cluster_col]}

    res = model.fit(**fit_kwargs)
    return ModelResults(
        model_type="ordinal",
        results=res,
        predictors=predictors,
        independent_vars=existing_indep_vars,
        marginal_effects=compute_ordinal_margeff(res, predictors),
    )

def fit_ols_model(
    df: pd.DataFrame,
    independent_vars: List[str],
    outcome_col: str = "neutrality_assessment",
    cluster_col: Optional[str] = CLUSTER_COLUMN,
    model_col: str = "Model",
    source_col: str = "Source",
    use_source: bool = True,
    use_model: bool = False,
    use_model_dep_interactions: bool = True,
    include_model_term: bool = True,
    source_cell_means: bool = False,
    control_vars: Optional[List[str]] = None,
    sum_code_controls: Optional[List[str]] = None,
) -> Optional[ModelResults]:
    """
    Fit OLS model. Automatically detects if independent vars are categorical
    (bool, object, category) or continuous (int, float).

    `control_vars` are added as plain main-effect covariates (no interactions).
    Categorical controls listed in `sum_code_controls` are sum/deviation-coded
    (`C(var, Sum)`) so their coefficients are deviations from the grand mean
    rather than contrasts against a dropped baseline level.

    source_cell_means=True (requires use_source=True) reparameterises the
    source term with full-rank dummies and no overall intercept, so that the
    per-source ideology slopes appear as stand-alone coefficients (no
    reference level). Overrides `use_model_dep_interactions`.
    """
    # Filter to dep_vars that exist in df
    existing_indep_vars = [dv for dv in independent_vars if dv in df.columns]

    # Helper to decide if a variable needs C() wrapper
    def needs_categorical_encoding(col_name: str) -> bool:
        dtype = df[col_name].dtype
        # Treat object, category, and bool as categorical
        if pd.api.types.is_object_dtype(dtype): return True
        if pd.api.types.is_categorical_dtype(dtype): return True
        if pd.api.types.is_bool_dtype(dtype): return True
        # Treat numeric (int, float) as continuous
        return False


    # 1. Build Main Effects Terms
    dep_var_terms_list = []
    for dv in existing_indep_vars:
        if needs_categorical_encoding(dv):
            dep_var_terms_list.append(f"C({dv})")
        else:
            dep_var_terms_list.append(f"{dv}")
    dep_var_terms = " + ".join(dep_var_terms_list)

    # 2. Build Interaction Terms (encoded to match each dep var)
    model_dep_interactions = " + ".join(
        f"C({model_col}):{t}" for t in dep_var_terms_list
    )
    source_dep_interactions = " + ".join(
        f"C({source_col}):{t}" for t in dep_var_terms_list
    )

    # 3. Assemble Formula
    if use_source and source_cell_means:
        # Cell-means parameterisation: no overall intercept, full-rank
        # dummies for source, and source-interacted ideology terms -> each
        # source gets its own slope row.
        if include_model_term:
            formula = (
                f"{outcome_col} ~ 0 + C({source_col}) + C({model_col}) "
                f"+ C({source_col}):({dep_var_terms})"
            )
        else:
            formula = (
                f"{outcome_col} ~ 0 + C({source_col}) "
                f"+ C({source_col}):({dep_var_terms})"
            )
    elif use_source:
        if not include_model_term:
            formula = (
                f"{outcome_col} ~ C({source_col}) + {dep_var_terms} + "
                f"{source_dep_interactions}"
            )
        else:
            model_source_interaction = f"C({model_col}):C({source_col})"
            if use_model_dep_interactions:
                formula = (
                    f"{outcome_col} ~ C({model_col}) + C({source_col}) + {dep_var_terms} + "
                    f"{model_dep_interactions} + {source_dep_interactions} + {model_source_interaction}"
                )
            else:
                formula = (
                    f"{outcome_col} ~ C({model_col}) + C({source_col}) + {dep_var_terms} + "
                    f"{source_dep_interactions} + {model_source_interaction}"
                )
    elif use_model:
        if use_model_dep_interactions:
            formula = (
                f"{outcome_col} ~ C({model_col}) + {dep_var_terms} + {model_dep_interactions}"
            )
        else:
            formula = f"{outcome_col} ~ C({model_col}) + {dep_var_terms}"
    else:
        formula = f"{outcome_col} ~ {dep_var_terms}"

    # Plain main-effect controls (no interactions), appended to whatever RHS the
    # branch above produced.
    if control_vars:
        sum_coded = set(sum_code_controls or [])

        def control_term(cv: str) -> str:
            if not needs_categorical_encoding(cv):
                return f"{cv}"
            return f"C({cv}, Sum)" if cv in sum_coded else f"C({cv})"

        control_terms = " + ".join(
            control_term(cv) for cv in control_vars if cv in df.columns
        )
        if control_terms:
            formula = f"{formula} + {control_terms}"

    print(f"Generated Formula:\n{formula}\n")

    model = smf.ols(formula, data=df)
    if cluster_col is not None and cluster_col in df.columns:
        res = model.fit(cov_type=OLS_COV_TYPE, cov_kwds={"groups": df[cluster_col]})
    else:
        res = model.fit()

    anova_table = anova_lm(res, typ=2)

    return ModelResults(
        model_type="ols",
        results=res,
        anova_table=anova_table,
        independent_vars=existing_indep_vars,
    )

def fit_all_models(
    df: pd.DataFrame,
    independent_vars: List[str],
    outcome_col: str = "neutrality_assessment",
    cluster_col: Optional[str] = CLUSTER_COLUMN,
    model_col: str = "Model",
    source_col: str = "Source",
    use_source: bool = True,
    use_model: bool = False,
    use_model_dep_interactions: bool = True,
    include_model_term: bool = True,
    source_cell_means: bool = False,
    control_vars: Optional[List[str]] = None,
    ols_control_vars: Optional[List[str]] = None,
    sum_code_controls: Optional[List[str]] = None,
) -> AllModelResults:
    """Fit both ordinal and OLS models with all dependent variables as predictors.

    `control_vars` are added to both models; `ols_control_vars` are added to the
    OLS model only (e.g. high-cardinality fixed effects that are too slow/unstable
    in the ordinal fit). Categorical controls named in `sum_code_controls` are
    sum/deviation-coded in the OLS fit (no dropped baseline level).
    """
    ols_controls = list(control_vars or []) + list(ols_control_vars or [])
    return AllModelResults(
        ordinal=fit_ordinal_model(df, independent_vars, outcome_col, cluster_col, model_col, source_col, use_source, use_model, use_model_dep_interactions, include_model_term, control_vars=control_vars),
        ols=fit_ols_model(df, independent_vars, outcome_col, cluster_col, model_col, source_col, use_source, use_model, use_model_dep_interactions, include_model_term, source_cell_means, control_vars=(ols_controls or None), sum_code_controls=sum_code_controls),
    )


def print_model_summary(model_results: ModelResults, table_dir: Path) -> None:
    """Print formatted model summary."""
    dep_str = ""
    if model_results.independent_vars:
        dep_str = f" [dep_vars: {', '.join(model_results.independent_vars)}]"
    summary_text = str(model_results.results.summary())

    print(f"\n{'=' * 80}")
    print(f"{model_results.model_type.upper()} REGRESSION{dep_str}")
    print(f"{'=' * 80}\n")
    print(summary_text)

    output_parts = [
        f"{'=' * 80}",
        f"{model_results.model_type.upper()} REGRESSION{dep_str}",
        f"{'=' * 80}",
        "",
        summary_text,
    ]

    if model_results.anova_table is not None:
        print(f"\n{'=' * 80}")
        print("ANOVA TABLE (Type II)")
        print(f"{'=' * 80}\n")
        print(model_results.anova_table)
        output_parts.extend(
            [
                "",
                f"{'=' * 80}",
                "ANOVA TABLE (Type II)",
                f"{'=' * 80}",
                "",
                model_results.anova_table.to_string(),
            ]
        )

    if hasattr(model_results.results, 'tvalues') or hasattr(model_results.results, 'pvalues'):
        results = model_results.results
        test_stats = results.tvalues if hasattr(results, 'tvalues') else results.pvalues
        pvals = results.pvalues

        # Build Wald test table
        wald_data = {
            'Variable': results.params.index.tolist(),
            'Test Stat': test_stats.values,
            'P-value': pvals.values,
        }

        import pandas as pd
        wald_df = pd.DataFrame(wald_data)

        # Add significance stars
        wald_df['Signif'] = wald_df['P-value'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        )

        print(f"\n{'=' * 80}")
        print("WALD TESTS (Individual Coefficients)")
        print(f"{'=' * 80}\n")
        print(wald_df.to_string(index=False))

        output_parts.extend(
            [
                "",
                f"{'=' * 80}",
                "WALD TESTS (Individual Coefficients)",
                f"{'=' * 80}",
                "",
                wald_df.to_string(index=False),
            ]
        )

    table_dir.parent.mkdir(parents=True, exist_ok=True)
    table_dir.write_text("\n".join(output_parts) + "\n", encoding="utf-8")
    print()


def print_all_model_summaries(all_results: AllModelResults, table_dir: Path) -> None:
    """Print summaries for both models."""
    if all_results.ordinal:
        print_model_summary(all_results.ordinal, table_dir=table_dir / "ordinal_model_summary.txt")
    if all_results.ols:
        print_model_summary(all_results.ols, table_dir=table_dir / "ols_model_summary.txt")

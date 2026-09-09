import numpy as np
import pandas as pd
import pytest

from sportsmodel.model.trueprob import (
    FEATURES,
    feature_matrix,
    fit_ridge,
    fit_model,
    predict,
    residual_sigma,
)

SEED = 7
N = 400
NOISE_SD = 1.5


def _synthetic_df(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    noise = rng.normal(scale=NOISE_SD, size=n)
    margin = 3 * a - 2 * b + noise
    signal = 3 * a - 2 * b
    return pd.DataFrame({"a": a, "b": b, "margin": margin, "signal": signal})


def test_features_module_constant_is_nfl_ordered_list():
    assert FEATURES == [
        "elo_diff",
        "last10_diff",
        "sos_diff",
        "sov_diff",
        "rest_diff",
        "off_epa_diff",
        "def_epa_diff",
    ]


def test_fit_model_predict_recovers_signal_on_holdout():
    df = _synthetic_df()
    split = 300
    train, test = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)

    model = fit_model(train, "margin", ["a", "b"], alpha=1.0)
    preds = predict(model, test)

    assert preds.shape == (len(test),)

    corr = np.corrcoef(preds, test["signal"].to_numpy())[0, 1]
    assert corr > 0.9

    mae = np.mean(np.abs(preds - test["margin"].to_numpy()))
    # noise sd is 1.5; a good fit should have MAE well under 2x that.
    assert mae < 2.0


def test_feature_matrix_train_computes_mu_sd_and_imputes_to_train_mean():
    df = _synthetic_df()
    X, mu, sd = feature_matrix(df, ["a", "b"])
    assert X.shape == (len(df), 2)
    np.testing.assert_allclose(mu, df[["a", "b"]].mean().to_numpy())
    np.testing.assert_allclose(sd, df[["a", "b"]].std(ddof=0).to_numpy(), rtol=1e-6, atol=1e-8)
    # standardized train columns should have ~mean 0
    np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-8)


def test_nan_feature_imputed_to_train_mean_and_predict_does_not_crash():
    df = _synthetic_df()
    split = 300
    train, test = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)

    model = fit_model(train, "margin", ["a", "b"], alpha=1.0)

    test_with_nan = test.copy()
    test_with_nan.loc[0, "a"] = np.nan

    preds = predict(model, test_with_nan)
    assert np.isfinite(preds).all()

    # Manually build expectation: NaN 'a' imputed to train mean -> standardized 'a' becomes 0
    # for that row, so prediction should equal intercept + coef_b * standardized_b.
    X_full, mu, sd = feature_matrix(test_with_nan, ["a", "b"], model["mu"], model["sd"])
    assert X_full[0, 0] == 0.0  # (train_mean - train_mean) / sd == 0


def test_fit_ridge_large_alpha_shrinks_non_intercept_coefficients():
    df = _synthetic_df()
    X, mu, sd = feature_matrix(df, ["a", "b"])
    y = df["margin"].to_numpy()

    coef_small_alpha = fit_ridge(X, y, alpha=0.0)
    coef_large_alpha = fit_ridge(X, y, alpha=1e8)

    small_sum = np.sum(np.abs(coef_small_alpha[1:]))
    large_sum = np.sum(np.abs(coef_large_alpha[1:]))
    assert large_sum < small_sum * 1e-3

    # Intercept stays close to mean(y) regardless of penalty on the slopes.
    assert coef_large_alpha[0] == pytest.approx(np.mean(y), abs=1e-2)


def test_residual_sigma_matches_synthetic_noise_std():
    df = _synthetic_df()
    y_true = df["margin"].to_numpy()
    y_pred = df["signal"].to_numpy()
    sigma = residual_sigma(y_true, y_pred)
    assert sigma == pytest.approx(NOISE_SD, rel=0.15)


def test_rows_with_nan_target_are_dropped_before_fitting():
    df = _synthetic_df()
    df_with_nan_target = df.copy()
    df_with_nan_target.loc[0, "margin"] = np.nan

    # Should not raise and should not let a NaN leak into the fitted coefficients.
    model = fit_model(df_with_nan_target, "margin", ["a", "b"], alpha=1.0)
    assert np.isfinite(model["coef"]).all()

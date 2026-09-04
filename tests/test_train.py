import pytest
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

from house_price_mlops.train import build_tuned_model, get_models, get_tuning_configs


def test_get_models_contains_ten_baselines():
    models = get_models()
    assert len(models) == 10
    assert {"Lasso", "Gradient Boosting", "Hist Gradient Boosting"}.issubset(models)


def test_tuning_configs_are_exactly_three_candidates():
    configs = get_tuning_configs()
    assert set(configs) == {"Lasso", "Gradient Boosting", "Hist Gradient Boosting"}
    for config in configs.values():
        assert "estimator" in config
        assert "param_grid" in config
        assert config["param_grid"]


def test_build_tuned_lasso_uses_params_from_search():
    model = build_tuned_model("Lasso", {"model__alpha": 0.003})
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["model"], Lasso)
    assert model.named_steps["model"].alpha == pytest.approx(0.003)


def test_build_tuned_gradient_boosting_uses_search_params():
    model = build_tuned_model(
        "Gradient Boosting",
        {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 300},
    )
    assert model.learning_rate == pytest.approx(0.05)
    assert model.max_depth == 4
    assert model.n_estimators == 300


def test_unknown_tuned_model_raises():
    with pytest.raises(KeyError):
        build_tuned_model("Unknown", {})

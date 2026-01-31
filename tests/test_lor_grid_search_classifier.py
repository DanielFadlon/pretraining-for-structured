import textwrap

import numpy as np
from sklearn.datasets import make_classification

from src.classifiers.lor_grid_search_classifier import LoRGridSearchClassifier


def test_yaml_loader_adds_random_state_for_saga(tmp_path):
    cfg_path = tmp_path / "lor_grid.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """\
            binary:
              - solver: ["saga"]
                penalty: ["l1"]
                C: [1.0]
                multi_class: ["ovr"]
            multiclass: []
            """
        ),
        encoding="utf-8",
    )

    clf = LoRGridSearchClassifier(
        hyperparams=str(cfg_path),
        scoring="accuracy",
        num_classes=1,
        cv=2,
        n_jobs=1,
        random_state=123,
        max_iter=200,
    )

    assert isinstance(clf.grid_search.param_grid, list)
    assert clf.grid_search.param_grid[0]["clf__solver"] == ["saga"]
    assert clf.grid_search.param_grid[0]["clf__random_state"] == [123]


def test_binary_fit_predict_from_yaml(tmp_path):
    cfg_path = tmp_path / "lor_grid.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """\
            binary:
              - solver: ["liblinear"]
                penalty: ["l2"]
                C: [1.0]
                multi_class: ["ovr"]
            multiclass:
              - solver: ["lbfgs"]
                penalty: ["l2"]
                C: [1.0]
                multi_class: ["multinomial"]
            """
        ),
        encoding="utf-8",
    )

    X, y = make_classification(
        n_samples=80,
        n_features=10,
        n_informative=6,
        n_redundant=0,
        random_state=0,
    )

    clf = LoRGridSearchClassifier(
        hyperparams=str(cfg_path),
        scoring="accuracy",
        num_classes=1,
        cv=2,
        n_jobs=1,
        random_state=0,
        max_iter=200,
    ).fit(X, y)

    preds = clf.predict(X)
    probs = clf.predict_probs(X)

    assert preds.shape == (X.shape[0],)
    assert probs.shape == (X.shape[0],)
    assert np.all((0.0 <= probs) & (probs <= 1.0))


def test_multiclass_fit_predict_from_yaml(tmp_path):
    cfg_path = tmp_path / "lor_grid.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """\
            binary:
              - solver: ["liblinear"]
                penalty: ["l2"]
                C: [1.0]
                multi_class: ["ovr"]
            multiclass:
              - solver: ["lbfgs"]
                penalty: ["l2"]
                C: [1.0]
                multi_class: ["multinomial"]
            """
        ),
        encoding="utf-8",
    )

    X, y = make_classification(
        n_samples=120,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )

    clf = LoRGridSearchClassifier(
        hyperparams=str(cfg_path),
        scoring="accuracy",
        num_classes=3,
        cv=2,
        n_jobs=1,
        random_state=0,
        max_iter=300,
    ).fit(X, y)

    preds = clf.predict(X)
    probs = clf.predict_probs(X)

    assert preds.shape == (X.shape[0],)
    assert probs.shape == (X.shape[0], 3)
    assert np.all((0.0 <= probs) & (probs <= 1.0))


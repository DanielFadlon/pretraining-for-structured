import numpy as np
import joblib
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from src.utils.file_utils import read_yaml

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Default config path (relative to project root)
DEFAULT_HYPERPARAMS_PATH = "configs/classifiers/lor_hyperparams.yml"


def _load_hyperparams_from_yaml(yaml_path: str, num_classes: int, random_state: int) -> list[dict]:
    """Load hyperparams from YAML and convert to sklearn GridSearchCV format."""
    config = read_yaml(yaml_path)

    key = "binary" if num_classes in (0, 1) else "multiclass"
    raw_params = config[key]

    # Convert to sklearn param_grid format (add 'clf__' prefix)
    param_grid = []
    for params in raw_params:
        grid_entry = {f"clf__{k}": v for k, v in params.items()}
        # Add random_state for saga solver
        if "saga" in params.get("solver", []):
            grid_entry["clf__random_state"] = [random_state]
        param_grid.append(grid_entry)

    return param_grid


class LoRGridSearchClassifier:
    """
    GridSearch wrapper for (penalized) Logistic Regression.

    Parameters
    ----------
    hyperparams : dict | list[dict] | str | None
        - If dict/list[dict]: Used directly as GridSearchCV param_grid.
        - If str: Path to YAML config file.
        - If None: Loads from default YAML config.
    scoring : str | None
        Sklearn scorer. If None, defaults to:
          - 'roc_auc' for binary
          - 'roc_auc_ovr_weighted' for multiclass
    num_classes : int
        Number of classes. Use 1 for binary, or >1 for multiclass.
    class_weight : 'balanced' | dict | None
        Passed to LogisticRegression. 'balanced' is often good if labels are skewed.
    cv : int
        Cross-validation folds.
    n_jobs : int
        Parallelism for GridSearchCV.
    random_state : int | None
        For reproducibility.
    max_iter : int
        Maximum iterations for LogisticRegression solver.
    """

    def __init__(
        self,
        hyperparams=None,
        scoring=None,
        num_classes=1,
        class_weight="balanced",
        cv=5,
        n_jobs=-1,
        random_state=42,
        max_iter=2000,
    ) -> None:
        self.num_classes = num_classes
        self.random_state = random_state

        # Default scoring
        if scoring is None:
            scoring = "roc_auc" if num_classes in (0, 1) else "roc_auc_ovr_weighted"

        # Base estimator inside a pipeline with scaling
        lr = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
        )
        estimator = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", lr),
        ])

        # Load hyperparams
        param_grid = self._resolve_hyperparams(hyperparams)

        self.best_model = None
        self.grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
            verbose=0,
        )

    def _resolve_hyperparams(self, hyperparams) -> list[dict]:
        """Resolve hyperparams from various input types."""
        if hyperparams is None:
            # Load from default YAML
            return _load_hyperparams_from_yaml(
                DEFAULT_HYPERPARAMS_PATH, self.num_classes, self.random_state
            )
        elif isinstance(hyperparams, str):
            # Load from specified YAML path
            return _load_hyperparams_from_yaml(
                hyperparams, self.num_classes, self.random_state
            )
        else:
            # Use directly (dict or list[dict])
            return hyperparams


    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        if self.num_classes in (0, 1):
            # Ensure binary labels are 0/1 for ROC AUC stability
            unique = np.unique(y_train)
            if set(unique) not in (set([0, 1]), set([0.0, 1.0])):
                # Map smallest to 0, largest to 1
                mapping = {cls: idx for idx, cls in enumerate(sorted(unique))}
                y_train = np.vectorize(mapping.get)(y_train)
        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_
        return self

    def save_best_model(self, model_path: str):
        if self.best_model is None:
            raise Exception("Train first")
        joblib.dump(self.best_model, f"{model_path}.joblib")

    def load_best_model(self, model_path: str):
        self.best_model = joblib.load(f"{model_path}.joblib")
        return self

    def predict(self, X: np.ndarray):
        if self.best_model is None:
            raise Exception("Train first or load a model")
        return self.best_model.predict(X)

    def predict_probs(self, X: np.ndarray):
        if self.best_model is None:
            raise Exception("Train first or load a model")
        proba = self.best_model.predict_proba(X)
        if self.num_classes in (0, 1):
            return proba[:, 1]
        return proba

    def best_params_(self):
        if self.grid_search.best_params_ is None:
            raise Exception("Train first")
        return self.grid_search.best_params_

    def __str__(self):
        res = f"""
                -------------- Best Model -----------------

                    {str(self.best_model)}

                -------------------------------------------
        """
        return res

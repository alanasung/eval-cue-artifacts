"""Shared fixtures for the whole suite.

Three rules govern everything in ``tests/``:

1. **No weights are ever downloaded.** ``tiny_model`` builds a randomly
   initialised GPT-2 with 2 layers and a 32-dim residual stream from a config
   object, so it never touches the network. Anything that genuinely needs real
   weights is marked ``slow`` and deselected in CI.
2. **No test writes outside ``tmp_path``.** Run directories, caches, and result
   JSONs all live under a per-test temporary directory.
3. **Synthetic data used to validate the harness carries a planted answer.** The
   ``planted_failure_case`` fixture below is not random noise: it is constructed
   so that the correct output of the failure harness is known by hand, which is
   what makes it a test of the instrument rather than a smoke test.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Import-path setup ─────────────────────────────────────────────────────────
# The package is normally pip-installed (`pip install -e .`) so imports resolve
# without help. Adding src/ keeps the suite runnable from a fresh clone that has
# not been installed yet, which is the state CI is in before `pip install -e .`
# and the state a reviewer is in when they first clone.

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

CONFIG_ROOT = REPO_ROOT / "configs"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: needs model weights or a long run; deselected by default in CI"
    )


# ── Paths and run directories ─────────────────────────────────────────────────


@pytest.fixture
def repo_root():
    return REPO_ROOT


@pytest.fixture
def config_root():
    """The repository's Hydra config tree, or a skip if it is absent."""
    if not CONFIG_ROOT.is_dir():
        pytest.skip(f"no config tree at {CONFIG_ROOT}")
    return CONFIG_ROOT


@pytest.fixture
def run_dir(tmp_path):
    """A run directory in the shape the runner produces, under tmp_path."""
    path = tmp_path / "runs" / "2026-01-01_00-00-00_test"
    (path / "artifacts").mkdir(parents=True)
    return path


@pytest.fixture
def cache_root(tmp_path):
    path = tmp_path / "artifacts"
    path.mkdir()
    return path


# ── Config ────────────────────────────────────────────────────────────────────


@pytest.fixture
def pilot_config(tmp_path):
    """A pilot-profile Config sized so that any test using it finishes fast."""
    from evalaware.configs.schema import Config

    cfg = Config()
    cfg.paths.root = str(tmp_path)
    cfg.paths.runs = str(tmp_path / "runs")
    cfg.paths.cache = str(tmp_path / "cache")
    cfg.paths.results = str(tmp_path / "results")
    cfg.paths.figures = str(tmp_path / "results" / "figures")
    cfg.paths.tables = str(tmp_path / "results" / "tables")
    cfg.data.n_items = 8
    cfg.model.batch_size = 2
    cfg.model.max_new_tokens = 4
    cfg.eval.bootstrap_samples = 200
    cfg.experiment.name = "pilot_fixture"
    cfg.experiment.task_id = "E00"
    return cfg


@pytest.fixture
def fixture_config_dir(tmp_path):
    """A minimal but genuine Hydra config tree with a real base_experiment.

    Composition is tested against the repository's own ``configs/`` elsewhere.
    This tree exists so the *loader* can be tested in isolation from whatever
    presets happen to be checked in.
    """
    root = tmp_path / "hydra_configs"
    (root / "experiment").mkdir(parents=True)
    (root / "model").mkdir()

    # ``base_schema`` is the typed Config registered by loader.register_schema().
    # Composing on top of it is what makes a misspelled key fail here rather than
    # resolving to None three stages later.
    (root / "config.yaml").write_text(
        "defaults:\n"
        "  - base_schema\n"
        "  - experiment: alpha\n"
        "  - model: tiny\n"
        "  - _self_\n"
        "\n"
        "run:\n"
        "  seed: 0\n",
        encoding="utf-8",
    )
    (root / "model" / "tiny.yaml").write_text(
        "# @package model\nname: sshleifer/tiny-gpt2\nbatch_size: 2\n", encoding="utf-8"
    )
    (root / "experiment" / "base_experiment.yaml").write_text(
        "# @package _global_\n"
        "experiment:\n"
        "  name: base\n"
        "  task_id: E00\n"
        "  description: Shared base every preset inherits\n"
        "  stages: []\n"
        "eval:\n"
        "  threshold: 0.5\n"
        "  tolerance: 0.1\n",
        encoding="utf-8",
    )
    for name, task_id in (("alpha", "E01"), ("beta", "E02")):
        (root / "experiment" / f"{name}.yaml").write_text(
            "# @package _global_\n"
            "defaults:\n"
            "  - base_experiment\n"
            "  - _self_\n"
            "\n"
            "experiment:\n"
            f"  name: {name}\n"
            f"  task_id: {task_id}\n",
            encoding="utf-8",
        )
    return root


# ── Models ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def tiny_model_config():
    """GPT-2 config small enough to instantiate in milliseconds."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    return transformers.GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=64,
        n_positions=32,
        n_ctx=32,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )


@pytest.fixture
def tiny_model(tiny_model_config):
    """A real but minuscule GPT-2, randomly initialised.

    Randomly initialised is the point: it exercises exactly the same module
    tree, hook points, and forward signature as a real checkpoint while
    downloading nothing. ``eval()`` and zeroed dropout make forward passes
    deterministic so hook tests can compare logits directly.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(tiny_model_config)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@pytest.fixture
def tiny_input_ids():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 64, (2, 8), generator=generator)


@pytest.fixture
def echo_llm():
    """An LLM backend that is a pure function, for wrapper tests."""

    def call(prompt):
        return f"echo::{prompt}"

    return call


# ── Synthetic data with planted structure ─────────────────────────────────────

# One row per example. Chosen so every quantity the failure harness reports can
# be recomputed by hand from these twelve numbers.
PLANTED_TARGET = [0.05, 0.15, 0.25, 0.35, 0.45, 0.48, 0.55, 0.62, 0.72, 0.82, 0.92, 0.98]
PLANTED_THRESHOLD = 0.5
PLANTED_TOLERANCE = 0.1
PLANTED_MARGIN = 0.2


class PlantedCase:
    """A failure-analysis problem whose correct answer is known in advance.

    The three components are planted with distinct, deliberately *dissociated*
    roles, because the whole point of the harness is that value and silence are
    different axes:

    ``a`` contributes almost nothing (dropping it moves predictions by 0.01).
    ``b`` is the most valuable (largest leave-one-out MAE increase) but fails
    **loudly**: the two decisions it flips land within 0.05 of the boundary.
    ``c`` is less valuable than ``b`` but fails **silently**: the two decisions
    it flips land 0.25 away from the boundary, confidently wrong.

    A harness that conflated the two axes would name the same component twice.
    """

    components = ["a", "b", "c"]
    threshold = PLANTED_THRESHOLD
    tolerance = PLANTED_TOLERANCE
    confidence_margin = PLANTED_MARGIN

    # Known answers, all derived by hand in tests/test_failure_harness.py.
    # ``most_silent_component`` is reported under the dropout condition key, so
    # the expected value is the condition name rather than the bare component.
    expected_most_valuable = "b"
    expected_most_silent = "drop_c"
    expected_full_mae = 0.0
    expected_drop_a_mae = 0.01
    expected_drop_b_mae = 1.57 / 12
    expected_drop_c_mae = 0.57 / 12
    expected_best_solo_mae = 0.12
    expected_winners = {"a": 6, "b": 4, "c": 2}

    def __init__(self):
        self.target = np.array(PLANTED_TARGET, dtype=float)
        self.gate_true = self.target >= self.threshold
        self.n = self.target.size

        full = self.target.copy()

        drop_a = self.target + 0.01

        drop_b = self.target.copy()
        drop_b[5] = 0.55  # 0.48 -> decision flips up, 0.05 from the boundary: loud
        drop_b[6] = 0.45  # 0.55 -> decision flips down, 0.05 from the boundary: loud
        drop_b[0] = 0.40  # decision still correct, error 0.35: imprecise
        drop_b[1] = 0.45  # imprecise
        drop_b[10] = 0.55  # imprecise
        drop_b[11] = 0.60  # imprecise

        drop_c = self.target.copy()
        drop_c[5] = 0.75  # flips up and sits 0.25 from the boundary: silent
        drop_c[6] = 0.25  # flips down and sits 0.25 from the boundary: silent

        solo_a = np.full(self.n, 0.5)
        solo_c = np.full(self.n, 0.5)
        solo_b = 0.5 * self.target + 0.25  # shrunk halfway to the boundary

        self.table = {
            frozenset({"a", "b", "c"}): full,
            frozenset({"b", "c"}): drop_a,
            frozenset({"a", "c"}): drop_b,
            frozenset({"a", "b"}): drop_c,
            frozenset({"a"}): solo_a,
            frozenset({"b"}): solo_b,
            frozenset({"c"}): solo_c,
        }

    def predict_fn(self, present):
        """A ``PredictFn``: a pure lookup, so nothing is left to chance."""
        key = frozenset(present)
        if key not in self.table:
            raise AssertionError(f"harness asked for an unplanned subset: {sorted(key)}")
        return self.table[key].copy()

    def groups(self):
        """A stratification attribute with two balanced levels."""
        return {"half": np.array(["lo"] * 6 + ["hi"] * 6)}


@pytest.fixture
def planted():
    return PlantedCase()


@pytest.fixture
def linear_planted_factory():
    """Build a randomised two-component problem with a planted value ordering.

    Unlike :class:`PlantedCase` the numbers here are random, but the *ordering*
    is planted: ``strong`` carries three times the signal of ``weak``, so a
    working harness must rank its marginal value higher at every seed. This is
    the recovery-rate test, run across many seeds.
    """

    def build(seed, n=200):
        rng = np.random.default_rng(seed)
        strong = rng.normal(size=n)
        weak = rng.normal(size=n)
        target = 0.75 * strong + 0.25 * weak

        signals = {"strong": strong, "weak": weak}
        weights = {"strong": 0.75, "weak": 0.25}

        def predict_fn(present):
            # Absent components are mean-imputed, which for a centred signal
            # means they contribute nothing.
            out = np.zeros(n)
            for name, weight in weights.items():
                if name in present:
                    out = out + weight * signals[name]
            return out

        return {
            "components": ["strong", "weak"],
            "target": target,
            "gate_true": target >= 0.0,
            "predict_fn": predict_fn,
            "expected_most_valuable": "strong",
        }

    return build


@pytest.fixture
def binary_scores():
    """Scores and labels with a hand-checkable AUROC of exactly 0.75.

    Positives sit at ranks 2 and 4 of four items, so of the four
    positive/negative pairs three are correctly ordered: 3/4 = 0.75.
    """
    return {
        "scores": np.array([0.1, 0.2, 0.3, 0.4]),
        "labels": np.array([False, True, False, True]),
        "expected_auroc": 0.75,
    }


@pytest.fixture
def result_rows():
    """Aggregation input: three arms measured on the same items."""
    return [
        {"task": "E01_baseline", "seed": 0, "metrics": {"accuracy": 0.60, "silent_rate": 0.30}},
        {"task": "E01_baseline", "seed": 1, "metrics": {"accuracy": 0.70, "silent_rate": 0.20}},
        {"task": "E02_treatment", "seed": 0, "metrics": {"accuracy": 0.80, "silent_rate": 0.10}},
    ]

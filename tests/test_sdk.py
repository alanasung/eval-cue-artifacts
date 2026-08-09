"""API-contract / SDK integrity tests."""
from __future__ import annotations
import importlib
import pkgutil
import evalaware as pkg

def test_version():
    assert hasattr(pkg, "__version__")

_SHARED_TOP = {
    "ablation",
    "cache",
    "configs",
    "data",
    "evaluation",
    "experiments",
    "models",
    "reporting",
    "utils",
}


def test_all_submodules_import():
    """Import every shared-spine submodule.

    Domain packages (``src/<pkg>/<pkg>/``) and mid-build stage modules
    (``stages``, ``runner``) are excluded — parallel domain agents may leave
    those half-wired while the shared spine stays import-clean.
    """
    prefix = pkg.__name__ + "."
    pkg_name = pkg.__name__
    domain_prefix = f"{pkg_name}.{pkg_name.split('.')[-1]}."
    failures = []
    for mod in pkgutil.walk_packages(pkg.__path__, prefix):
        relative = mod.name[len(prefix) :]
        top = relative.split(".", 1)[0]
        if mod.name.startswith(domain_prefix):
            continue
        if top not in _SHARED_TOP:
            continue
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001 - collect all
            failures.append(f"{mod.name}: {exc}")
    assert not failures, failures

def test_public_evaluation_surface():
    from evalaware import evaluation
    for name in ("accuracy", "auroc", "EvaluationHarness", "PredictFn"):
        assert hasattr(evaluation, name)

def test_public_cache_surface():
    from evalaware.cache import ArtifactCache
    assert callable(ArtifactCache)

def test_reporting_surface():
    from evalaware import reporting
    for name in ("aggregate_results", "write_tables", "write_figures"):
        assert hasattr(reporting, name)

def test_hooks_surface():
    from evalaware.models import hooks
    for name in ("capture", "steer", "ablate", "patch_activations", "intervene_attention_heads"):
        assert hasattr(hooks, name)
        assert callable(getattr(hooks, name))

def test_generation_chat_helper():
    from evalaware.models.generation import apply_chat_template
    assert callable(apply_chat_template)

def test_registry_has_revision():
    from evalaware.models.registry import smallest
    spec = smallest()
    assert getattr(spec, "revision", None)

def test_no_print_in_library():
    import pathlib

    root = pathlib.Path(pkg.__file__).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "print(" in stripped:
                offenders.append(f"{path.relative_to(root)}:{i}:{stripped}")
    assert offenders == []

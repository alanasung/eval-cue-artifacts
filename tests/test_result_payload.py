"""The self-describing result contract.

A result file has to explain itself: which task, which seed, which commit, over
how many examples, and how many rows were dropped. These tests exist to keep
that contract from being quietly relaxed.
"""

import json

import pytest

from evalaware.utils.run_manifest import ResultPayload, save_results

REQUIRED_PROVENANCE = ("task", "seed", "git_sha", "n", "metrics", "created_at")


def make_payload(**overrides):
    kwargs = {
        "task": "E01_baseline",
        "seed": 0,
        "n": {"train": 60, "val": 20, "test": 20},
        "metrics": {"accuracy": 0.5},
    }
    kwargs.update(overrides)
    return ResultPayload(**kwargs)


class TestConstruction:
    def test_minimal_payload_constructs(self):
        assert make_payload().task == "E01_baseline"

    def test_profile_defaults_to_pilot(self):
        assert make_payload().profile == "pilot"

    def test_git_sha_is_captured_without_being_asked(self):
        assert isinstance(make_payload().git_sha, str)

    def test_created_at_is_captured_without_being_asked(self):
        assert make_payload().created_at

    @pytest.mark.parametrize("field", ["inputs", "notes"])
    def test_list_fields_default_to_empty(self, field):
        assert getattr(make_payload(), field) == []

    def test_list_defaults_are_not_shared(self):
        first, second = make_payload(), make_payload()
        first.notes.append("mine")
        assert second.notes == []

    def test_model_is_optional(self):
        assert make_payload().model is None
        assert make_payload(model="gpt2").model == "gpt2"


class TestValidation:
    def test_empty_task_is_rejected(self):
        with pytest.raises(ValueError, match="must be a non-empty task id"):
            make_payload(task="")

    def test_empty_task_message_shows_the_expected_shape(self):
        with pytest.raises(ValueError, match=r"E01_baseline"):
            make_payload(task="")

    @pytest.mark.parametrize("n_dropped", [-1, -10])
    def test_negative_drop_count_is_rejected(self, n_dropped):
        with pytest.raises(ValueError, match="n_dropped must be non-negative"):
            make_payload(n_dropped=n_dropped)

    def test_dropping_rows_without_a_reason_is_rejected(self):
        with pytest.raises(ValueError, match="silent data loss is not permitted"):
            make_payload(n_dropped=3)

    def test_drop_message_names_the_count(self):
        with pytest.raises(ValueError, match=r"^3 rows were dropped"):
            make_payload(n_dropped=3)

    def test_dropping_rows_with_a_reason_is_allowed(self):
        payload = make_payload(n_dropped=3, dropped_reason="non-finite activations")
        assert payload.n_dropped == 3

    def test_zero_drops_needs_no_reason(self):
        assert make_payload(n_dropped=0).dropped_reason is None

    @pytest.mark.parametrize("n_dropped", [1, 2, 50])
    def test_any_positive_drop_count_demands_a_reason(self, n_dropped):
        with pytest.raises(ValueError):
            make_payload(n_dropped=n_dropped)


class TestToDict:
    def test_contains_every_provenance_field(self):
        payload = make_payload().to_dict()
        for key in REQUIRED_PROVENANCE:
            assert key in payload, f"result payload lost provenance field {key!r}"

    def test_is_json_serializable(self):
        json.dumps(make_payload().to_dict())

    def test_split_sizes_survive(self):
        assert make_payload().to_dict()["n"] == {"train": 60, "val": 20, "test": 20}

    def test_measured_results_carry_no_warning(self):
        assert "WARNING" not in make_payload().to_dict()

    def test_synthetic_results_are_loudly_marked(self):
        payload = make_payload(is_synthetic=True).to_dict()
        assert "SYNTHETIC DATA" in payload["WARNING"]

    def test_synthetic_marker_is_machine_checkable(self):
        payload = make_payload(is_synthetic=True).to_dict()
        assert payload["is_synthetic"] is True
        assert "not a measured result" in payload["WARNING"]

    def test_drop_accounting_is_reported_not_hidden(self):
        payload = make_payload(n_dropped=2, dropped_reason="NaN embeddings").to_dict()
        assert payload["n_dropped"] == 2
        assert payload["dropped_reason"] == "NaN embeddings"


class TestSaveResults:
    def test_saves_a_payload_object(self, tmp_path):
        path = save_results(tmp_path / "results.json", make_payload())
        assert json.loads(path.read_text(encoding="utf-8"))["task"] == "E01_baseline"

    def test_creates_parent_directories(self, tmp_path):
        assert save_results(tmp_path / "deep" / "results.json", make_payload()).is_file()

    def test_leaves_no_temp_files(self, tmp_path):
        save_results(tmp_path / "results.json", make_payload())
        assert list(tmp_path.glob("*.tmp")) == []

    def test_accepts_a_mapping_that_carries_provenance(self, tmp_path):
        raw = {"task": "E02", "seed": 1, "git_sha": "abc", "metrics": {}}
        path = save_results(tmp_path / "r.json", raw)
        assert json.loads(path.read_text(encoding="utf-8"))["task"] == "E02"

    @pytest.mark.parametrize(
        ("raw", "missing"),
        [
            ({"seed": 0, "git_sha": "a"}, "task"),
            ({"task": "E01", "git_sha": "a"}, "seed"),
            ({"task": "E01", "seed": 0}, "git_sha"),
        ],
    )
    def test_bare_dicts_missing_provenance_are_rejected(self, tmp_path, raw, missing):
        with pytest.raises(TypeError, match=missing):
            save_results(tmp_path / "r.json", raw)

    def test_rejection_message_points_at_resultpayload(self, tmp_path):
        with pytest.raises(TypeError, match="construct a ResultPayload instead"):
            save_results(tmp_path / "r.json", {"metrics": {}})

    def test_empty_dict_is_rejected(self, tmp_path):
        with pytest.raises(TypeError, match="missing provenance fields"):
            save_results(tmp_path / "r.json", {})

    def test_synthetic_marker_survives_the_round_trip(self, tmp_path):
        path = save_results(tmp_path / "r.json", make_payload(is_synthetic=True))
        assert "WARNING" in json.loads(path.read_text(encoding="utf-8"))

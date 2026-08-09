"""Run directories, atomic JSON writes, and run metadata."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from evalaware.utils.run_manifest import (
    RunMetadata,
    create_run_dir,
    save_run_metadata,
    utc_now_iso,
    write_json_atomic,
)


class TestUtcNowIso:
    def test_parses_as_iso8601(self):
        datetime.fromisoformat(utc_now_iso())

    def test_is_timezone_aware(self):
        assert datetime.fromisoformat(utc_now_iso()).tzinfo is not None

    def test_is_utc(self):
        assert datetime.fromisoformat(utc_now_iso()).utcoffset().total_seconds() == 0

    def test_is_monotonic_enough_to_order(self):
        assert utc_now_iso() <= utc_now_iso()


class TestCreateRunDir:
    def test_creates_the_directory(self, tmp_path):
        assert create_run_dir(tmp_path / "runs").is_dir()

    def test_creates_missing_parents(self, tmp_path):
        created = create_run_dir(tmp_path / "deep" / "nested" / "runs")
        assert created.is_dir()

    def test_name_is_a_parseable_timestamp(self, tmp_path):
        created = create_run_dir(tmp_path / "runs")
        datetime.strptime(created.name, "%Y-%m-%d_%H-%M-%S")

    def test_suffix_is_appended(self, tmp_path):
        assert create_run_dir(tmp_path / "runs", suffix="probe").name.endswith("_probe")

    def test_suffix_keeps_the_timestamp_prefix(self, tmp_path):
        created = create_run_dir(tmp_path / "runs", suffix="probe")
        datetime.strptime(created.name.rsplit("_probe", 1)[0], "%Y-%m-%d_%H-%M-%S")

    def test_same_second_collisions_get_distinct_directories(self, tmp_path):
        base = tmp_path / "runs"
        created = [create_run_dir(base, suffix="x") for _ in range(5)]
        assert len({p.name for p in created}) == 5

    def test_collision_suffixes_are_numeric(self, tmp_path):
        base = tmp_path / "runs"
        first = create_run_dir(base, suffix="x")
        second = create_run_dir(base, suffix="x")
        assert second.name.startswith(first.name)
        assert second.name[len(first.name) :].lstrip("-").isdigit()

    def test_never_reuses_an_existing_directory(self, tmp_path):
        base = tmp_path / "runs"
        first = create_run_dir(base)
        (first / "marker.txt").write_text("x", encoding="utf-8")
        second = create_run_dir(base)
        assert not (second / "marker.txt").exists()

    def test_accepts_a_string_base(self, tmp_path):
        assert create_run_dir(str(tmp_path / "runs")).is_dir()

    def test_returns_a_path(self, tmp_path):
        assert isinstance(create_run_dir(tmp_path / "runs"), Path)


class TestWriteJsonAtomic:
    def test_round_trips(self, tmp_path):
        path = write_json_atomic(tmp_path / "out.json", {"a": 1, "b": [2, 3]})
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": [2, 3]}

    def test_creates_parent_directories(self, tmp_path):
        path = write_json_atomic(tmp_path / "a" / "b" / "c.json", {"k": "v"})
        assert path.is_file()

    def test_leaves_no_temp_files_behind(self, tmp_path):
        write_json_atomic(tmp_path / "out.json", {"k": "v"})
        assert list(tmp_path.glob("*.tmp")) == []

    def test_overwrites_cleanly(self, tmp_path):
        path = tmp_path / "out.json"
        write_json_atomic(path, {"version": 1})
        write_json_atomic(path, {"version": 2})
        assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}

    def test_non_serializable_values_fall_back_to_str(self, tmp_path):
        path = write_json_atomic(tmp_path / "out.json", {"when": datetime(2026, 1, 1)})
        assert json.loads(path.read_text(encoding="utf-8"))["when"].startswith("2026-01-01")

    def test_is_indented_for_human_review(self, tmp_path):
        path = write_json_atomic(tmp_path / "out.json", {"a": 1})
        assert "\n  " in path.read_text(encoding="utf-8")

    def test_a_failed_write_leaves_no_partial_file(self, tmp_path, monkeypatch):
        import json as json_module

        path = tmp_path / "out.json"

        def explode(*_args, **_kwargs):
            raise RuntimeError("simulated kill during write")

        monkeypatch.setattr(json_module, "dump", explode)
        with pytest.raises(RuntimeError, match="simulated kill"):
            write_json_atomic(path, {"a": 1})
        assert not path.exists()
        assert list(tmp_path.glob("*.tmp")) == []


class TestRunMetadata:
    def test_to_dict_contains_every_provenance_field(self):
        record = RunMetadata(
            script="scripts/run.py",
            seed=3,
            profile="pilot",
            git_sha="abc",
            git_branch="main",
            git_dirty=False,
        )
        payload = record.to_dict()
        for key in ("script", "seed", "profile", "git_sha", "git_branch", "git_dirty"):
            assert key in payload

    def test_created_at_is_populated_automatically(self):
        record = RunMetadata("s", 0, "pilot", "abc", "main", False)
        datetime.fromisoformat(record.created_at)

    @pytest.mark.parametrize(
        "field", ["hardware", "packages", "determinism", "extra"]
    )
    def test_optional_dict_fields_default_to_empty(self, field):
        record = RunMetadata("s", 0, "pilot", "abc", "main", False)
        assert getattr(record, field) == {}

    def test_defaults_are_not_shared_between_instances(self):
        first = RunMetadata("s", 0, "pilot", "abc", "main", False)
        second = RunMetadata("s", 0, "pilot", "abc", "main", False)
        first.extra["only_mine"] = True
        assert second.extra == {}

    def test_to_dict_is_json_serializable(self):
        json.dumps(RunMetadata("s", 0, "pilot", "abc", "main", False).to_dict())


class TestSaveRunMetadata:
    def test_writes_the_expected_filename(self, tmp_path):
        assert save_run_metadata(tmp_path, {"note": "x"}).name == "run_metadata.json"

    def test_caller_metadata_lands_under_extra(self, tmp_path):
        path = save_run_metadata(tmp_path, {"device": "cpu", "batch_size": 4})
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["extra"] == {"device": "cpu", "batch_size": 4}

    def test_git_provenance_is_attached_without_being_asked(self, tmp_path):
        payload = json.loads(save_run_metadata(tmp_path, {}).read_text(encoding="utf-8"))
        assert "git_sha" in payload
        assert "git_branch" in payload
        assert isinstance(payload["git_dirty"], bool)

    @pytest.mark.parametrize(
        ("seed", "profile", "script"),
        [(0, "pilot", "a.py"), (7, "full", "b.py"), (99, "pilot", "scripts/c.py")],
    )
    def test_run_identity_is_recorded(self, tmp_path, seed, profile, script):
        path = save_run_metadata(tmp_path, {}, seed=seed, profile=profile, script=script)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["seed"] == seed
        assert payload["profile"] == profile
        assert payload["script"] == script

    def test_defaults_are_conservative(self, tmp_path):
        payload = json.loads(save_run_metadata(tmp_path, {}).read_text(encoding="utf-8"))
        assert payload["seed"] == 0
        assert payload["profile"] == "pilot"
        assert payload["script"] == "unknown"

    def test_creates_the_run_directory_if_absent(self, tmp_path):
        assert save_run_metadata(tmp_path / "brand" / "new", {}).is_file()

    def test_rewriting_replaces_rather_than_appends(self, tmp_path):
        save_run_metadata(tmp_path, {"pass": 1})
        path = save_run_metadata(tmp_path, {"pass": 2})
        assert json.loads(path.read_text(encoding="utf-8"))["extra"] == {"pass": 2}

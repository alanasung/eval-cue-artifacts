"""Artifact cache round-trip, introspection, and the resume path."""

import json

import numpy as np
import pytest

from evalaware.cache.artifact_cache import ArtifactCache, CacheRecord


@pytest.fixture
def cache(cache_root):
    return ArtifactCache(cache_root, "residuals", version="v1")


@pytest.fixture
def sample():
    return np.arange(12, dtype=np.float32).reshape(3, 4)


class TestRoundTrip:
    def test_write_then_read_returns_the_same_values(self, cache, sample):
        cache.write("rec0", sample)
        assert np.array_equal(np.asarray(cache.read("rec0")), sample)

    def test_dtype_survives(self, cache, sample):
        cache.write("rec0", sample)
        assert cache.read("rec0").dtype == sample.dtype

    def test_shape_survives(self, cache, sample):
        cache.write("rec0", sample)
        assert cache.read("rec0").shape == sample.shape

    @pytest.mark.parametrize(
        "array",
        [
            np.array([1.0]),
            np.zeros((2, 2)),
            np.arange(6).reshape(1, 6),
            np.linspace(0, 1, 10).astype(np.float64),
            np.ones((2, 3, 4), dtype=np.int32),
        ],
    )
    def test_round_trips_many_shapes_and_dtypes(self, cache, array):
        cache.write("rec", array)
        assert np.array_equal(np.asarray(cache.read("rec")), array)

    def test_read_without_mmap_returns_a_plain_array(self, cache, sample):
        cache.write("rec0", sample)
        assert isinstance(cache.read("rec0", mmap=False), np.ndarray)

    def test_getitem_is_read(self, cache, sample):
        cache.write("rec0", sample)
        assert np.array_equal(np.asarray(cache["rec0"]), sample)

    def test_write_returns_a_record_describing_the_array(self, cache, sample):
        record = cache.write("rec0", sample)
        assert isinstance(record, CacheRecord)
        assert record.id == "rec0"
        assert record.namespace == "residuals"
        assert record.version == "v1"
        assert record.shape == [3, 4]
        assert record.dtype == "float32"

    def test_files_land_under_the_namespace(self, cache, cache_root, sample):
        cache.write("rec0", sample)
        assert (cache_root / "residuals" / "rec0.npy").is_file()

    def test_writes_leave_no_temp_files(self, cache, cache_root, sample):
        cache.write("rec0", sample)
        assert list((cache_root / "residuals").glob("*.tmp")) == []

    def test_namespaces_are_isolated(self, cache_root, sample):
        ArtifactCache(cache_root, "left", version="v1").write("shared", sample)
        right = ArtifactCache(cache_root, "right", version="v1")
        assert not right.has("shared")


class TestWriteGuards:
    def test_existing_id_is_not_silently_overwritten(self, cache, sample):
        cache.write("rec0", sample)
        with pytest.raises(FileExistsError, match="already cached"):
            cache.write("rec0", sample)

    def test_overwrite_message_names_the_flag(self, cache, sample):
        cache.write("rec0", sample)
        with pytest.raises(FileExistsError, match="pass overwrite=True to replace"):
            cache.write("rec0", sample)

    def test_overwrite_replaces_the_contents(self, cache, sample):
        cache.write("rec0", sample)
        cache.write("rec0", sample * 2, overwrite=True)
        assert np.array_equal(np.asarray(cache.read("rec0")), sample * 2)

    def test_empty_arrays_are_refused(self, cache):
        with pytest.raises(ValueError, match="expected a non-empty numeric array"):
            cache.write("empty", np.array([]))

    def test_object_arrays_are_refused(self, cache):
        with pytest.raises(ValueError, match="expected a non-empty numeric array"):
            cache.write("objects", np.array([{"a": 1}, {"b": 2}], dtype=object))

    def test_refusal_message_names_the_record(self, cache):
        with pytest.raises(ValueError, match="refusing to cache 'bad'"):
            cache.write("bad", np.array([]))

    def test_missing_key_read_raises_keyerror(self, cache):
        with pytest.raises(KeyError, match="is not cached"):
            cache.read("never-written")

    def test_missing_key_message_suggests_has(self, cache):
        with pytest.raises(KeyError, match="call has\\(\\) first"):
            cache.read("never-written")


class TestResumePath:
    def test_has_is_false_before_writing(self, cache):
        assert cache.has("rec0") is False

    def test_has_is_true_after_writing(self, cache, sample):
        cache.write("rec0", sample)
        assert cache.has("rec0") is True

    def test_contains_operator_matches_has(self, cache, sample):
        cache.write("rec0", sample)
        assert "rec0" in cache
        assert "rec1" not in cache

    def test_missing_returns_everything_on_a_cold_cache(self, cache):
        ids = ["a", "b", "c"]
        assert cache.missing(ids) == ids

    def test_missing_returns_nothing_on_a_warm_cache(self, cache, sample):
        for rid in ["a", "b", "c"]:
            cache.write(rid, sample)
        assert cache.missing(["a", "b", "c"]) == []

    def test_missing_returns_only_the_gap(self, cache, sample):
        cache.write("a", sample)
        cache.write("c", sample)
        assert cache.missing(["a", "b", "c", "d"]) == ["b", "d"]

    def test_missing_preserves_input_order(self, cache, sample):
        cache.write("b", sample)
        assert cache.missing(["z", "y", "b", "x"]) == ["z", "y", "x"]

    def test_missing_of_an_empty_list_is_empty(self, cache):
        assert cache.missing([]) == []

    def test_resume_loop_only_recomputes_the_gap(self, cache, sample):
        """The whole point of has()/missing(): a crash must not cost the run."""
        ids = [f"rec{i}" for i in range(6)]
        for rid in ids[:4]:
            cache.write(rid, sample)

        computed = []
        for rid in cache.missing(ids):
            computed.append(rid)
            cache.write(rid, sample)

        assert computed == ["rec4", "rec5"]
        assert cache.missing(ids) == []

    def test_a_second_pass_computes_nothing(self, cache, sample):
        ids = ["a", "b"]
        for rid in ids:
            cache.write(rid, sample)
        assert list(cache.missing(ids)) == []


class TestIntrospection:
    def test_keys_are_empty_before_anything_is_written(self, cache):
        assert cache.keys() == []

    def test_keys_are_sorted(self, cache, sample):
        for rid in ["delta", "alpha", "charlie", "bravo"]:
            cache.write(rid, sample)
        assert cache.keys() == ["alpha", "bravo", "charlie", "delta"]

    def test_keys_come_from_disk_not_the_manifest(self, cache, sample):
        """A manifest truncated by a crash must not hide a cached array."""
        cache.write("rec0", sample)
        cache.manifest_path.write_text("", encoding="utf-8")
        assert cache.keys() == ["rec0"]
        assert cache.has("rec0")

    def test_len_counts_cached_arrays(self, cache, sample):
        assert len(cache) == 0
        cache.write("a", sample)
        cache.write("b", sample)
        assert len(cache) == 2

    def test_iteration_yields_keys(self, cache, sample):
        cache.write("a", sample)
        cache.write("b", sample)
        assert sorted(cache) == ["a", "b"]

    def test_disk_usage_is_zero_on_a_cold_cache(self, cache):
        assert cache.disk_usage_bytes() == 0

    def test_disk_usage_grows_with_writes(self, cache, sample):
        cache.write("a", sample)
        first = cache.disk_usage_bytes()
        cache.write("b", sample)
        assert cache.disk_usage_bytes() > first

    def test_disk_usage_excludes_the_manifest(self, cache, sample):
        cache.write("a", sample)
        assert cache.disk_usage_bytes() == (cache.dir / "a.npy").stat().st_size


class TestManifestLines:
    def test_no_records_before_writing(self, cache):
        assert cache.records() == []

    def test_one_line_per_write(self, cache, sample):
        cache.write("a", sample)
        cache.write("b", sample)
        assert len(cache.records()) == 2

    def test_manifest_is_append_only_jsonl(self, cache, sample):
        cache.write("a", sample)
        cache.write("b", sample)
        lines = cache.manifest_path.read_text(encoding="utf-8").strip().splitlines()
        assert [json.loads(line)["id"] for line in lines] == ["a", "b"]

    def test_records_carry_provenance(self, cache, sample):
        cache.write("a", sample, timestamp="2026-01-01T00:00:00+00:00")
        record = cache.records()[0]
        assert record.version == "v1"
        assert record.timestamp == "2026-01-01T00:00:00+00:00"
        assert record.path == "a.npy"

    def test_truncated_lines_are_skipped_not_fatal(self, cache, sample):
        cache.write("a", sample)
        with cache.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write('{"id": "b", "namespa')
        assert [r.id for r in cache.records()] == ["a"]

    def test_blank_lines_are_skipped(self, cache, sample):
        cache.write("a", sample)
        with cache.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n")
        assert len(cache.records()) == 1

    def test_lines_missing_required_keys_are_skipped(self, cache, sample):
        cache.write("a", sample)
        with cache.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": "b"}) + "\n")
        assert [r.id for r in cache.records()] == ["a"]

    def test_record_dict_round_trip(self):
        record = CacheRecord(
            id="x",
            namespace="ns",
            version="v1",
            timestamp="2026-01-01T00:00:00+00:00",
            shape=[2, 3],
            dtype="float32",
            path="x.npy",
        )
        assert CacheRecord.from_dict(record.to_dict()) == record

    def test_record_to_dict_keys(self):
        record = CacheRecord("x", "ns", "v1", "t", [1], "float32", "x.npy")
        assert set(record.to_dict()) == {
            "id",
            "namespace",
            "version",
            "timestamp",
            "shape",
            "dtype",
            "path",
        }

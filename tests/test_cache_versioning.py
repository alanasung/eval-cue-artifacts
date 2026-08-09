"""Version pinning and path-traversal defences on the artifact cache.

Two activation sets from different model revisions silently mixing into one
probe fit is the bug this module exists to make impossible, so the
version-mismatch path gets as much attention as the happy path.
"""

import numpy as np
import pytest

from evalaware.cache.artifact_cache import ArtifactCache, check_safe_name


@pytest.fixture
def sample():
    return np.arange(6, dtype=np.float32)


class TestVersionPinning:
    def test_first_write_pins_the_version(self, cache_root, sample):
        ArtifactCache(cache_root, "ns", version="v1").write("a", sample)
        assert (cache_root / "ns" / "version.txt").read_text(encoding="utf-8") == "v1"

    def test_same_version_reopens_cleanly(self, cache_root, sample):
        ArtifactCache(cache_root, "ns", version="v1").write("a", sample)
        again = ArtifactCache(cache_root, "ns", version="v1")
        assert np.array_equal(np.asarray(again.read("a")), sample)

    def test_write_under_a_different_version_raises(self, cache_root, sample):
        ArtifactCache(cache_root, "ns", version="v1").write("a", sample)
        with pytest.raises(ValueError, match="version mismatch"):
            ArtifactCache(cache_root, "ns", version="v2").write("b", sample)

    def test_read_under_a_different_version_raises(self, cache_root, sample):
        ArtifactCache(cache_root, "ns", version="v1").write("a", sample)
        with pytest.raises(ValueError, match="version mismatch"):
            ArtifactCache(cache_root, "ns", version="v2").read("a")

    def test_mismatch_message_names_both_versions(self, cache_root, sample):
        ArtifactCache(cache_root, "ns", version="encoder-v1").write("a", sample)
        with pytest.raises(ValueError, match=r"cache holds 'encoder-v1'"):
            ArtifactCache(cache_root, "ns", version="encoder-v2").read("a")

    def test_mismatch_message_names_the_producing_version(self, cache_root, sample):
        ArtifactCache(cache_root, "ns", version="encoder-v1").write("a", sample)
        with pytest.raises(ValueError, match=r"this process produces 'encoder-v2'"):
            ArtifactCache(cache_root, "ns", version="encoder-v2").read("a")

    def test_mismatch_message_states_the_three_fixes(self, cache_root, sample):
        ArtifactCache(cache_root, "ns", version="v1").write("a", sample)
        with pytest.raises(ValueError) as excinfo:
            ArtifactCache(cache_root, "ns", version="v2").read("a")
        message = str(excinfo.value)
        assert "Re-run with the original version" in message
        assert "different cache root" in message
        assert "clear the namespace" in message

    @pytest.mark.parametrize(
        ("pinned", "requested"),
        [
            ("v1", "v2"),
            ("v1", "V1"),
            ("abc123", "abc124"),
            ("v1", "unknown"),
            ("2026-01-01", "2026-01-02"),
        ],
    )
    def test_any_difference_is_a_mismatch(self, cache_root, sample, pinned, requested):
        ArtifactCache(cache_root, "ns", version=pinned).write("a", sample)
        with pytest.raises(ValueError, match="version mismatch"):
            ArtifactCache(cache_root, "ns", version=requested).read("a")

    def test_lax_reads_do_not_raise(self, cache_root, sample):
        ArtifactCache(cache_root, "ns", version="v1").write("a", sample)
        lax = ArtifactCache(cache_root, "ns", version="v2", strict_version=False)
        assert np.array_equal(np.asarray(lax.read("a")), sample)

    def test_lax_writes_still_enforce(self, cache_root, sample):
        ArtifactCache(cache_root, "ns", version="v1").write("a", sample)
        lax = ArtifactCache(cache_root, "ns", version="v2", strict_version=False)
        with pytest.raises(ValueError, match="version mismatch"):
            lax.write("b", sample)

    def test_a_separate_root_sidesteps_the_pin(self, tmp_path, sample):
        ArtifactCache(tmp_path / "one", "ns", version="v1").write("a", sample)
        other = ArtifactCache(tmp_path / "two", "ns", version="v2")
        other.write("a", sample)
        assert other.has("a")

    def test_different_namespaces_pin_independently(self, cache_root, sample):
        ArtifactCache(cache_root, "left", version="v1").write("a", sample)
        ArtifactCache(cache_root, "right", version="v2").write("a", sample)
        assert (cache_root / "right" / "version.txt").read_text(encoding="utf-8") == "v2"

    def test_records_carry_the_version_that_produced_them(self, cache_root, sample):
        cache = ArtifactCache(cache_root, "ns", version="v1")
        cache.write("a", sample)
        assert {r.version for r in cache.records()} == {"v1"}

    def test_has_does_not_enforce_the_version(self, cache_root, sample):
        """Resume checks must work before a version decision is made."""
        ArtifactCache(cache_root, "ns", version="v1").write("a", sample)
        assert ArtifactCache(cache_root, "ns", version="v2").has("a") is True


class TestSafeNames:
    @pytest.mark.parametrize(
        "name", ["a", "rec0", "layer_12", "sample-001", "UPPER", "0", "x.y"]
    )
    def test_accepts_ordinary_names(self, name):
        assert check_safe_name(name, "record id") == name

    @pytest.mark.parametrize(
        "name", ["../escape", "a/b", "a\\b", ".", "..", "/absolute", "/", "nested/../x"]
    )
    def test_rejects_traversal_and_separators(self, name):
        with pytest.raises(ValueError, match="not a safe path component"):
            check_safe_name(name, "record id")

    def test_rejects_the_empty_string(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            check_safe_name("", "record id")

    def test_message_carries_the_caller_supplied_kind(self):
        with pytest.raises(ValueError, match="^namespace is not a safe"):
            check_safe_name("../x", "namespace")

    def test_message_states_the_rule(self):
        with pytest.raises(ValueError, match="must not contain path separators"):
            check_safe_name("a/b", "record id")

    @pytest.mark.parametrize("name", ["../escape", "a/b", ".."])
    def test_unsafe_namespace_is_rejected_at_construction(self, cache_root, name):
        with pytest.raises(ValueError, match="namespace"):
            ArtifactCache(cache_root, name, version="v1")

    @pytest.mark.parametrize("record_id", ["../escape", "sub/dir", ".."])
    def test_unsafe_record_id_is_rejected_on_write(self, cache_root, sample, record_id):
        cache = ArtifactCache(cache_root, "ns", version="v1")
        with pytest.raises(ValueError, match="record id"):
            cache.write(record_id, sample)

    @pytest.mark.parametrize("record_id", ["../escape", "sub/dir"])
    def test_unsafe_record_id_is_rejected_on_read(self, cache_root, record_id):
        cache = ArtifactCache(cache_root, "ns", version="v1")
        with pytest.raises(ValueError, match="record id"):
            cache.read(record_id)

    def test_unsafe_record_id_is_rejected_on_has(self, cache_root):
        cache = ArtifactCache(cache_root, "ns", version="v1")
        with pytest.raises(ValueError, match="record id"):
            cache.has("../escape")

    def test_nothing_is_written_outside_the_cache_root(self, tmp_path, sample):
        root = tmp_path / "cache"
        cache = ArtifactCache(root, "ns", version="v1")
        with pytest.raises(ValueError):
            cache.write("../../pwned", sample)
        assert not (tmp_path.parent / "pwned.npy").exists()
        assert not (tmp_path / "pwned.npy").exists()

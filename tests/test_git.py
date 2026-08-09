"""Git provenance helpers.

The defining property under test is that these never raise. A result-writing
path that dies because git is missing has traded a recoverable gap in
provenance for a lost run.
"""

import subprocess

import pytest

from evalaware.utils.git import GitState, git_is_dirty, git_sha, git_state


@pytest.fixture
def real_repo(tmp_path):
    """A genuine git repository with one commit, isolated from the caller's."""
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git is not installed")
    root = tmp_path / "repo"
    root.mkdir()
    env = {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "HOME": str(tmp_path),
    }
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True, env=env)
    (root / "tracked.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True, env=env)
    return root


@pytest.fixture
def bare_dir(tmp_path):
    """A directory that is definitively not inside any git repository."""
    path = tmp_path / "not_a_repo"
    path.mkdir()
    return path


class TestGitSha:
    def test_returns_a_string(self):
        assert isinstance(git_sha(), str)

    def test_full_sha_in_a_real_repo(self, real_repo):
        sha = git_sha(cwd=real_repo)
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_short_sha_is_a_prefix_of_the_full_sha(self, real_repo):
        assert git_sha(cwd=real_repo).startswith(git_sha(short=True, cwd=real_repo))

    def test_short_sha_is_shorter(self, real_repo):
        assert len(git_sha(short=True, cwd=real_repo)) < 40

    def test_unknown_outside_a_repository(self, bare_dir):
        assert git_sha(cwd=bare_dir) == "unknown"

    def test_does_not_raise_outside_a_repository(self, bare_dir):
        git_sha(cwd=bare_dir)
        git_sha(short=True, cwd=bare_dir)

    def test_is_stable_across_calls(self, real_repo):
        assert git_sha(cwd=real_repo) == git_sha(cwd=real_repo)


class TestGitIsDirty:
    def test_clean_tree_is_not_dirty(self, real_repo):
        assert git_is_dirty(cwd=real_repo) is False

    def test_modified_tracked_file_is_dirty(self, real_repo):
        (real_repo / "tracked.txt").write_text("two\n", encoding="utf-8")
        assert git_is_dirty(cwd=real_repo) is True

    def test_untracked_file_is_dirty(self, real_repo):
        (real_repo / "stray.txt").write_text("x\n", encoding="utf-8")
        assert git_is_dirty(cwd=real_repo) is True

    def test_outside_a_repository_reports_not_dirty(self, bare_dir):
        assert git_is_dirty(cwd=bare_dir) is False

    def test_returns_a_bool_not_a_string(self, real_repo):
        assert isinstance(git_is_dirty(cwd=real_repo), bool)


class TestGitState:
    def test_collects_all_three_fields(self, real_repo):
        state = git_state(cwd=real_repo)
        assert state.sha == git_sha(cwd=real_repo)
        assert state.branch == "main"
        assert state.dirty is False

    def test_clean_known_commit_is_reproducible(self, real_repo):
        assert git_state(cwd=real_repo).is_reproducible is True

    def test_dirty_tree_is_not_reproducible(self, real_repo):
        (real_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
        assert git_state(cwd=real_repo).is_reproducible is False

    def test_unknown_sha_is_not_reproducible(self, bare_dir):
        state = git_state(cwd=bare_dir)
        assert state.sha == "unknown"
        assert state.is_reproducible is False

    def test_unknown_branch_outside_a_repository(self, bare_dir):
        assert git_state(cwd=bare_dir).branch == "unknown"

    def test_to_dict_keys(self, real_repo):
        assert set(git_state(cwd=real_repo).to_dict()) == {
            "sha",
            "branch",
            "dirty",
            "is_reproducible",
        }

    def test_to_dict_is_json_serializable(self, real_repo):
        import json

        json.dumps(git_state(cwd=real_repo).to_dict())

    def test_state_is_frozen(self, real_repo):
        with pytest.raises(Exception):
            git_state(cwd=real_repo).sha = "0" * 40

    @pytest.mark.parametrize("dirty", [True, False])
    def test_reproducibility_requires_a_known_sha(self, dirty):
        assert GitState(sha="unknown", branch="main", dirty=dirty).is_reproducible is False

    @pytest.mark.parametrize(
        ("sha", "dirty", "expected"),
        [("abc123", False, True), ("abc123", True, False), ("unknown", False, False)],
    )
    def test_reproducibility_truth_table(self, sha, dirty, expected):
        assert GitState(sha=sha, branch="main", dirty=dirty).is_reproducible is expected

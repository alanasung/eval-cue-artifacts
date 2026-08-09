from evalaware.utils.git import git_sha

REQUIRED = ("task", "seed", "git_sha", "n")

def test_required_field_names():
    payload = {"task": "t", "seed": 0, "git_sha": git_sha(), "n": 1}
    for key in REQUIRED:
        assert key in payload

"""Seeding, determinism reporting, and worker-seed derivation."""

import os
import random

import numpy as np
import pytest

from evalaware.utils.reproducibility import (
    DeterminismReport,
    determinism_report,
    seed_everything,
    set_seed,
    worker_seed,
)


class TestSetSeed:
    def test_returns_a_report(self):
        report = set_seed(0)
        assert isinstance(report, DeterminismReport)
        assert report.seed == 0

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 12345, 2**31 - 1])
    def test_accepts_any_non_negative_seed(self, seed):
        assert set_seed(seed).seed == seed

    @pytest.mark.parametrize("seed", [-1, -2, -100])
    def test_rejects_negative_seed(self, seed):
        with pytest.raises(ValueError, match="seed must be non-negative"):
            set_seed(seed)

    def test_negative_seed_message_names_the_value_and_the_fix(self):
        with pytest.raises(ValueError, match=r"got -5; check the config value"):
            set_seed(-5)

    def test_python_random_is_reproducible(self):
        set_seed(123)
        first = [random.random() for _ in range(5)]
        set_seed(123)
        assert [random.random() for _ in range(5)] == first

    def test_numpy_legacy_global_is_reproducible(self):
        set_seed(123)
        first = np.random.rand(5)
        set_seed(123)
        assert np.array_equal(np.random.rand(5), first)

    def test_different_seeds_give_different_draws(self):
        set_seed(1)
        first = np.random.rand(10)
        set_seed(2)
        assert not np.array_equal(np.random.rand(10), first)

    def test_pythonhashseed_is_exported(self):
        set_seed(99)
        assert os.environ["PYTHONHASHSEED"] == "99"

    def test_torch_cpu_generator_is_reproducible(self):
        torch = pytest.importorskip("torch")
        set_seed(7)
        first = torch.randn(16)
        set_seed(7)
        assert torch.equal(torch.randn(16), first)

    def test_deterministic_flag_is_recorded(self):
        assert set_seed(0, deterministic=True).deterministic_requested is True
        assert set_seed(0, deterministic=False).deterministic_requested is False

    def test_non_deterministic_run_still_seeds_the_rngs(self):
        set_seed(5, deterministic=False)
        first = np.random.rand(4)
        set_seed(5, deterministic=False)
        assert np.array_equal(np.random.rand(4), first)


class TestDeterminismReport:
    def test_guarantees_always_name_python_and_numpy(self):
        report = set_seed(0)
        joined = " ".join(report.guarantees)
        assert "python random" in joined
        assert "numpy" in joined

    def test_sampling_caveat_is_always_present_when_torch_exists(self):
        pytest.importorskip("torch")
        report = set_seed(0)
        assert any("sampling-based generation" in c for c in report.caveats)

    def test_backend_is_one_of_the_known_values(self):
        assert set_seed(0).backend in {"cpu", "cuda", "mps", "none"}

    def test_to_dict_is_json_shaped(self):
        payload = set_seed(3).to_dict()
        assert set(payload) == {
            "seed",
            "deterministic_requested",
            "torch_available",
            "backend",
            "guarantees",
            "caveats",
        }
        assert isinstance(payload["guarantees"], list)
        assert isinstance(payload["caveats"], list)

    def test_to_dict_copies_the_lists(self):
        report = set_seed(3)
        payload = report.to_dict()
        payload["guarantees"].append("fabricated")
        assert "fabricated" not in report.guarantees

    def test_report_is_frozen(self):
        report = set_seed(0)
        with pytest.raises(Exception):
            report.seed = 1

    def test_standalone_report_does_not_touch_rng_state(self):
        set_seed(11)
        expected = np.random.rand(3)
        set_seed(11)
        determinism_report(999)
        assert np.array_equal(np.random.rand(3), expected)

    def test_standalone_report_carries_the_requested_seed(self):
        assert determinism_report(4321).seed == 4321

    def test_standalone_report_requests_determinism(self):
        assert determinism_report(0).deterministic_requested is True

    def test_mps_caveat_appears_only_on_mps(self):
        report = determinism_report(0)
        mentions_mps = any("MPS" in c for c in report.caveats)
        assert mentions_mps == (report.backend == "mps")


class TestSeedEverythingAlias:
    def test_alias_is_the_same_function(self):
        assert seed_everything is set_seed

    def test_alias_works(self):
        assert seed_everything(8).seed == 8


class TestWorkerSeed:
    @pytest.mark.parametrize(
        ("base", "worker", "expected"),
        [(0, 0, 0), (0, 1, 1), (1, 0, 1000), (1, 3, 1003), (42, 7, 42007)],
    )
    def test_hand_computed_values(self, base, worker, expected):
        assert worker_seed(base, worker) == expected

    def test_no_collisions_across_small_worker_counts(self):
        seeds = [worker_seed(base, worker) for base in range(5) for worker in range(16)]
        assert len(set(seeds)) == len(seeds)

    @pytest.mark.parametrize("worker_id", [-1, -5])
    def test_rejects_negative_worker_id(self, worker_id):
        with pytest.raises(ValueError, match="worker_id must be non-negative"):
            worker_seed(0, worker_id)

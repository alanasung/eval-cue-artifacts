"""The loud-versus-silent axis and the error taxonomy.

``silent_rate`` is the headline number: a component whose removal produces
confident, stable, wrong outputs is one no deployment monitor can catch. These
tests plant both kinds of failure and check the harness tells them apart.
"""

import numpy as np
import pytest

from evalaware.evaluation.failure import (
    CATEGORIES,
    analyze_component_failure,
    classify_taxonomy,
)


@pytest.fixture
def report(planted):
    return analyze_component_failure(
        planted.components,
        planted.target,
        planted.gate_true,
        planted.predict_fn,
        threshold=planted.threshold,
        tolerance=planted.tolerance,
        confidence_margin=planted.confidence_margin,
    )


class TestCategories:
    def test_three_categories(self):
        assert CATEGORIES == ("correct", "imprecise", "critical")

    @pytest.mark.parametrize("name", ["correct", "imprecise", "critical"])
    def test_each_category_is_named(self, name):
        assert name in CATEGORIES


class TestClassifyTaxonomy:
    def test_exact_predictions_are_correct(self):
        target = np.array([0.2, 0.8])
        cats = classify_taxonomy(
            target, target, target >= 0.5, threshold=0.5, tolerance=0.1
        )
        assert list(cats) == ["correct", "correct"]

    def test_a_small_error_on_the_right_side_is_correct(self):
        target = np.array([0.2])
        cats = classify_taxonomy(
            target, np.array([0.25]), np.array([False]), threshold=0.5, tolerance=0.1
        )
        assert cats[0] == "correct"

    def test_a_large_error_on_the_right_side_is_imprecise(self):
        target = np.array([0.05])
        cats = classify_taxonomy(
            target, np.array([0.45]), np.array([False]), threshold=0.5, tolerance=0.1
        )
        assert cats[0] == "imprecise"

    def test_a_decision_flip_is_critical(self):
        target = np.array([0.48])
        cats = classify_taxonomy(
            target, np.array([0.52]), np.array([False]), threshold=0.5, tolerance=0.1
        )
        assert cats[0] == "critical"

    def test_a_flip_beats_tolerance_even_when_the_error_is_tiny(self):
        """A 0.04 error that crosses the boundary is critical, not correct."""
        target = np.array([0.48])
        cats = classify_taxonomy(
            target, np.array([0.52]), np.array([False]), threshold=0.5, tolerance=0.5
        )
        assert cats[0] == "critical"

    @pytest.mark.parametrize(
        ("pred", "expected"),
        [
            (0.50, "correct"),
            (0.55, "correct"),
            (0.61, "correct"),
            (0.75, "imprecise"),
            (0.49, "critical"),
            (0.10, "critical"),
        ],
    )
    def test_boundary_table_for_a_positive_example(self, pred, expected):
        target = np.array([0.6])
        cats = classify_taxonomy(
            target, np.array([pred]), np.array([True]), threshold=0.5, tolerance=0.1
        )
        assert cats[0] == expected

    def test_the_threshold_itself_counts_as_positive(self):
        cats = classify_taxonomy(
            np.array([0.5]), np.array([0.5]), np.array([True]), threshold=0.5, tolerance=0.1
        )
        assert cats[0] == "correct"

    def test_tolerance_is_inclusive(self):
        cats = classify_taxonomy(
            np.array([0.6]), np.array([0.7]), np.array([True]), threshold=0.5, tolerance=0.1
        )
        assert cats[0] == "correct"

    def test_output_is_one_label_per_example(self, planted):
        cats = classify_taxonomy(
            planted.target,
            planted.target,
            planted.gate_true,
            threshold=0.5,
            tolerance=0.1,
        )
        assert cats.shape == planted.target.shape

    @pytest.mark.parametrize("tolerance", [-0.01, -1.0])
    def test_negative_tolerance_is_refused(self, tolerance, planted):
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            classify_taxonomy(
                planted.target,
                planted.target,
                planted.gate_true,
                threshold=0.5,
                tolerance=tolerance,
            )

    def test_shape_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            classify_taxonomy(
                np.array([0.1, 0.2]),
                np.array([0.1]),
                np.array([False, False]),
                threshold=0.5,
                tolerance=0.1,
            )

    def test_shape_mismatch_message_names_all_three_arrays(self):
        with pytest.raises(ValueError, match=r"target=\(2,\), pred=\(1,\), gate_true=\(2,\)"):
            classify_taxonomy(
                np.array([0.1, 0.2]),
                np.array([0.1]),
                np.array([False, False]),
                threshold=0.5,
                tolerance=0.1,
            )

    def test_shape_mismatch_message_states_the_rule(self):
        with pytest.raises(ValueError, match="one row per example"):
            classify_taxonomy(
                np.array([0.1, 0.2]),
                np.array([0.1]),
                np.array([False, False]),
                threshold=0.5,
                tolerance=0.1,
            )


class TestTaxonomyCounts:
    def test_full_condition_is_entirely_correct(self, report, planted):
        assert report.conditions["full"]["taxonomy"] == {
            "correct": planted.n,
            "imprecise": 0,
            "critical": 0,
        }

    def test_dropping_a_stays_within_tolerance(self, report):
        assert report.conditions["drop_a"]["taxonomy"] == {
            "correct": 12,
            "imprecise": 0,
            "critical": 0,
        }

    def test_dropping_b_produces_the_planted_mix(self, report):
        assert report.conditions["drop_b"]["taxonomy"] == {
            "correct": 6,
            "imprecise": 4,
            "critical": 2,
        }

    def test_dropping_c_produces_only_critical_errors(self, report):
        assert report.conditions["drop_c"]["taxonomy"] == {
            "correct": 10,
            "imprecise": 0,
            "critical": 2,
        }

    @pytest.mark.parametrize("condition", ["full", "drop_a", "drop_b", "drop_c"])
    def test_every_condition_accounts_for_every_example(self, report, condition, planted):
        assert sum(report.conditions[condition]["taxonomy"].values()) == planted.n

    @pytest.mark.parametrize("condition", ["full", "drop_a", "drop_b", "drop_c"])
    def test_every_condition_reports_an_mae(self, report, condition):
        assert isinstance(report.conditions[condition]["mae"], float)


class TestDropoutProfile:
    def test_a_harmless_drop_induces_nothing(self, report):
        assert report.dropout["drop_a"]["induced_critical"] == 0

    def test_no_induced_failures_means_no_rate(self, report):
        """A rate over zero events is not zero, it is undefined, and says so."""
        assert report.dropout["drop_a"]["silent_rate"] is None

    def test_the_loud_component_induces_two_failures(self, report):
        assert report.dropout["drop_b"]["induced_critical"] == 2

    def test_the_loud_component_fails_loudly(self, report):
        stats = report.dropout["drop_b"]
        assert stats["loud"] == 2
        assert stats["silent"] == 0
        assert stats["silent_rate"] == pytest.approx(0.0)

    def test_the_silent_component_induces_two_failures(self, report):
        assert report.dropout["drop_c"]["induced_critical"] == 2

    def test_the_silent_component_fails_silently(self, report):
        stats = report.dropout["drop_c"]
        assert stats["silent"] == 2
        assert stats["loud"] == 0
        assert stats["silent_rate"] == pytest.approx(1.0)

    @pytest.mark.parametrize("name", ["drop_a", "drop_b", "drop_c"])
    def test_loud_and_silent_partition_the_induced_failures(self, report, name):
        stats = report.dropout[name]
        assert stats["loud"] + stats["silent"] == stats["induced_critical"]

    @pytest.mark.parametrize(
        ("name", "expected"), [("drop_a", 0.01), ("drop_b", 1.57 / 12), ("drop_c", 0.57 / 12)]
    )
    def test_mean_output_shift_is_hand_computed(self, report, name, expected):
        assert report.dropout[name]["mean_output_shift"] == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        ("name", "expected"), [("drop_a", 0.01), ("drop_b", 1.57 / 12), ("drop_c", 0.57 / 12)]
    )
    def test_mae_increase_matches_the_marginal_value(self, report, name, expected):
        assert report.dropout[name]["mae_increase"] == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("name", ["drop_a", "drop_b", "drop_c"])
    def test_dropout_entries_have_a_stable_schema(self, report, name):
        assert set(report.dropout[name]) == {
            "induced_critical",
            "silent",
            "loud",
            "silent_rate",
            "mean_output_shift",
            "mae_increase",
        }


class TestMostSilentComponent:
    def test_recovers_the_planted_silent_component(self, report, planted):
        assert report.most_silent_component == planted.expected_most_silent

    def test_silence_is_not_the_same_as_value(self, report, planted):
        """The most valuable component is deliberately not the most silent."""
        assert not report.most_silent_component.endswith(planted.expected_most_valuable)

    def test_undefined_rates_are_ignored_rather_than_treated_as_zero(self, report):
        assert report.dropout["drop_a"]["silent_rate"] is None
        assert report.most_silent_component != "drop_a"

    def test_it_appears_in_the_payload(self, report, planted):
        assert report.to_dict()["most_silent_component"] == planted.expected_most_silent

    def test_no_induced_failures_anywhere_gives_none(self, planted):
        """A perfect model has no failures to classify, loud or silent."""
        target = planted.target

        def perfect(_present):
            return target.copy()

        result = analyze_component_failure(
            ["a", "b"],
            target,
            planted.gate_true,
            perfect,
            threshold=0.5,
            tolerance=0.1,
            confidence_margin=0.2,
        )
        assert result.most_silent_component is None


class TestConfidenceMargin:
    """The margin is the knob that decides what counts as monitorable."""

    def _run(self, planted, margin):
        return analyze_component_failure(
            planted.components,
            planted.target,
            planted.gate_true,
            planted.predict_fn,
            threshold=planted.threshold,
            tolerance=planted.tolerance,
            confidence_margin=margin,
        )

    def test_a_zero_margin_makes_everything_silent(self, planted):
        report = self._run(planted, 0.0)
        assert report.dropout["drop_b"]["silent"] == 2
        assert report.dropout["drop_c"]["silent"] == 2

    def test_a_huge_margin_makes_everything_loud(self, planted):
        report = self._run(planted, 10.0)
        assert report.dropout["drop_b"]["loud"] == 2
        assert report.dropout["drop_c"]["loud"] == 2

    @pytest.mark.parametrize("margin", [0.0, 0.05, 0.1, 0.2, 0.5])
    def test_the_induced_count_is_independent_of_the_margin(self, planted, margin):
        report = self._run(planted, margin)
        assert report.dropout["drop_b"]["induced_critical"] == 2
        assert report.dropout["drop_c"]["induced_critical"] == 2

    @pytest.mark.parametrize(
        ("margin", "expected"), [(0.0, 1.0), (0.04, 1.0), (0.06, 0.0), (0.1, 0.0), (1.0, 0.0)]
    )
    def test_the_loud_component_crosses_over_at_its_planted_distance(
        self, planted, margin, expected
    ):
        # drop_b's flipped decisions sit 0.05 from the boundary, so they read as
        # silent only while the margin stays below 0.05.
        report = self._run(planted, margin)
        assert report.dropout["drop_b"]["silent_rate"] == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("margin", "expected"), [(0.0, 1.0), (0.2, 1.0), (0.24, 1.0), (0.26, 0.0), (1.0, 0.0)]
    )
    def test_the_silent_component_crosses_over_at_its_planted_distance(
        self, planted, margin, expected
    ):
        # drop_c's flipped decisions sit 0.25 from the boundary.
        report = self._run(planted, margin)
        assert report.dropout["drop_c"]["silent_rate"] == pytest.approx(expected)

    def test_silence_requires_distance_from_the_boundary(self, planted):
        tight = self._run(planted, 0.26)
        assert tight.dropout["drop_c"]["silent"] == 0
        loose = self._run(planted, 0.24)
        assert loose.dropout["drop_c"]["silent"] == 2

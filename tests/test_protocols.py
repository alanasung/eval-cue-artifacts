"""The callable protocols that keep evaluation model-agnostic."""

import inspect
import typing

import numpy as np
import pytest

from evalaware.evaluation import protocols
from evalaware.evaluation.protocols import (
    ComponentSet,
    GenerateFn,
    JudgeFn,
    LLMBackend,
    PredictFn,
    ProbeFn,
    ScoreFn,
    SupportsToDict,
    as_component_set,
)

CALLABLE_ALIASES = [
    ("PredictFn", PredictFn),
    ("ScoreFn", ScoreFn),
    ("GenerateFn", GenerateFn),
    ("LLMBackend", LLMBackend),
    ("JudgeFn", JudgeFn),
    ("ProbeFn", ProbeFn),
]


class TestAliases:
    @pytest.mark.parametrize(("name", "alias"), CALLABLE_ALIASES)
    def test_alias_is_exported(self, name, alias):
        assert name in protocols.__all__

    @pytest.mark.parametrize(("name", "alias"), CALLABLE_ALIASES)
    def test_alias_is_a_callable_type(self, name, alias):
        assert typing.get_origin(alias) is not None

    def test_component_set_is_a_frozenset_type(self):
        assert typing.get_origin(ComponentSet) is frozenset

    def test_predict_fn_returns_an_array(self):
        assert typing.get_args(PredictFn)[-1] is np.ndarray

    def test_predict_fn_takes_a_component_set(self):
        assert typing.get_args(PredictFn)[0] == [ComponentSet]

    def test_llm_backend_is_string_to_string(self):
        assert typing.get_args(LLMBackend) == ([str], str)

    def test_generate_fn_is_batched(self):
        args, ret = typing.get_args(GenerateFn)
        assert args == [list[str]]
        assert ret == list[str]

    def test_judge_label_is_a_closed_set(self):
        assert set(typing.get_args(protocols.JudgeLabel)) == {
            "pass",
            "fail",
            "refuse",
            "unclear",
        }

    def test_judge_fn_adjudicates_a_prompt_and_a_completion(self):
        args, _ = typing.get_args(JudgeFn)
        assert args == [str, str]


class TestNoModelImports:
    """Nothing in evaluation/ may import a concrete model class."""

    def test_protocols_does_not_import_torch(self):
        source = inspect.getsource(protocols)
        assert "import torch" not in source

    def test_protocols_does_not_import_transformers(self):
        assert "transformers" not in inspect.getsource(protocols)

    def test_module_docstring_states_the_rule(self):
        assert "model-agnostic" in protocols.__doc__


class TestSupportsToDict:
    def test_an_object_with_to_dict_satisfies_the_protocol(self):
        class Payload:
            def to_dict(self):
                return {"a": 1}

        assert isinstance(Payload(), SupportsToDict)

    def test_an_object_without_to_dict_does_not(self):
        class Bare:
            pass

        assert not isinstance(Bare(), SupportsToDict)

    def test_a_plain_dict_does_not_satisfy_it(self):
        assert not isinstance({"a": 1}, SupportsToDict)

    def test_the_result_payload_satisfies_it(self):
        from evalaware.utils.run_manifest import ResultPayload

        payload = ResultPayload(task="E01", seed=0, n={"test": 1}, metrics={})
        assert isinstance(payload, SupportsToDict)


class TestAsComponentSet:
    @pytest.mark.parametrize(
        "value",
        [
            ["a", "b"],
            ("a", "b"),
            {"a", "b"},
            frozenset({"a", "b"}),
            iter(["a", "b"]),
        ],
    )
    def test_accepts_any_iterable_of_names(self, value):
        assert as_component_set(value) == frozenset({"a", "b"})

    def test_returns_a_frozenset(self):
        assert isinstance(as_component_set(["a"]), frozenset)

    def test_result_is_hashable_so_it_can_key_a_cache(self):
        {as_component_set(["a", "b"]): 1}

    def test_duplicates_collapse(self):
        assert as_component_set(["a", "a", "b"]) == frozenset({"a", "b"})

    def test_empty_is_allowed(self):
        assert as_component_set([]) == frozenset()

    def test_a_bare_string_is_rejected(self):
        with pytest.raises(TypeError, match="expected an iterable of component names"):
            as_component_set("ab")

    def test_the_string_error_suggests_the_fix(self):
        with pytest.raises(TypeError, match=r"wrap it as \{'ab'\}"):
            as_component_set("ab")

    @pytest.mark.parametrize("value", [1, 3.5, None, object()])
    def test_non_iterables_are_rejected(self, value):
        with pytest.raises(TypeError, match="components must be iterable"):
            as_component_set(value)

    def test_non_iterable_error_names_the_type(self):
        with pytest.raises(TypeError, match="got int"):
            as_component_set(7)

    @pytest.mark.parametrize("bad", [["a", ""], ["a", 1], ["a", None], [""], [b"a"]])
    def test_non_string_or_empty_names_are_rejected(self, bad):
        with pytest.raises(ValueError, match="component names must be non-empty strings"):
            as_component_set(bad)

    def test_the_name_error_shows_the_offending_value(self):
        with pytest.raises(ValueError, match="got 7"):
            as_component_set(["a", 7])


class TestProtocolsAreStructural:
    """Any plain function of the right shape is a valid implementation."""

    def test_a_lambda_is_a_predict_fn(self):
        predict = lambda present: np.zeros(len(present))  # noqa: E731
        assert predict(frozenset({"a", "b"})).shape == (2,)

    def test_a_dict_lookup_is_a_predict_fn(self, planted):
        assert planted.predict_fn(frozenset({"a", "b", "c"})).shape == (12,)

    def test_a_closure_is_an_llm_backend(self, echo_llm):
        assert echo_llm("hello") == "echo::hello"

    def test_a_probe_fn_maps_activations_to_scores(self):
        weights = np.array([1.0, -1.0, 0.5])

        def probe(activations):
            return activations @ weights

        assert probe(np.eye(3)).tolist() == [1.0, -1.0, 0.5]

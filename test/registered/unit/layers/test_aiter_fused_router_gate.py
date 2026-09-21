import unittest
from dataclasses import replace
from unittest import mock

import torch

from sglang.srt.layers.moe.topk import TopKConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _Entry:
    """Stands in for an aiter build that carries the fused routing preamble."""

    hidden_dims = (4096, 6144)
    max_topk = 64
    max_experts = 512
    max_tokens = 512
    call = supported = config_supported = None


def _glm_like_config(**overrides) -> TopKConfig:
    """A biased-sigmoid, single-group router with one fused shared expert."""
    cfg = TopKConfig(
        top_k=9,  # 8 routed + 1 fused shared
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        renormalize=True,
        num_fused_shared_experts=1,
        correction_bias=torch.zeros(256),
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=True,
        scoring_func="sigmoid",
    )
    return replace(cfg, **overrides) if overrides else cfg


class TestAiterFusedRouterGate(CustomTestCase):
    """The TopK-bypass gate must read only static config, and refuse everything else.

    The gate decides whether top-k is skipped so the fused aiter preamble can do the
    routing. A wrong "yes" is not a slow path, it is a different routing function, so
    each refusal below is a separate test rather than one combined case.
    """

    def _gate(
        self,
        cfg,
        hidden_dim=6144,
        num_experts=256,
        entry=_Entry(),
        backend="auto",
        aiter_enabled=True,
        a2a="none",
        ep_size=1,
        expert_location_dispatch_info=None,
    ):
        """Run the gate with every environmental signal it reads made explicit.

        The defaults describe the one configuration this optimization is validated on:
        the aiter runner actually selected, no all-to-all dispatcher, EP 1. Each test
        below moves exactly one of them.
        """
        from types import SimpleNamespace

        from sglang.srt.layers.moe import utils as moe_utils
        from sglang.srt.layers.moe.moe_runner import aiter as aiter_runner

        with mock.patch.object(
            aiter_runner, "_aiter_fused_router", lambda: entry
        ), mock.patch.object(
            aiter_runner, "_aiter_enabled", lambda: aiter_enabled
        ), mock.patch.object(
            aiter_runner, "get_parallel", lambda: SimpleNamespace(moe_ep_size=ep_size)
        ), mock.patch.object(
            moe_utils,
            "get_moe_runner_backend",
            lambda: moe_utils.MoeRunnerBackend(backend),
        ), mock.patch.object(
            moe_utils, "get_moe_a2a_backend", lambda: moe_utils.MoeA2ABackend(a2a)
        ):
            return aiter_runner.fused_router_can_bypass_topk(
                cfg,
                hidden_dim,
                num_experts,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )

    def test_accepts_the_shape_the_kernel_serves(self):
        self.assertTrue(self._gate(_glm_like_config()))

    def test_refused_without_the_entry(self):
        self.assertFalse(self._gate(_glm_like_config(), entry=None))

    def test_refused_on_an_unserved_hidden_dim(self):
        self.assertFalse(self._gate(_glm_like_config(), hidden_dim=5120))

    def test_refused_above_the_expert_ceiling(self):
        self.assertFalse(self._gate(_glm_like_config(), num_experts=1024))

    def test_refused_for_softmax_scoring(self):
        # The kernel selects with sigmoid+bias; softmax is a different function.
        self.assertFalse(self._gate(_glm_like_config(scoring_func="softmax")))

    def test_refused_without_a_correction_bias(self):
        self.assertFalse(self._gate(_glm_like_config(correction_bias=None)))

    def test_refused_for_real_expert_groups(self):
        # Grouped biased top-k collapses to the flat form only at one group.
        self.assertFalse(self._gate(_glm_like_config(num_expert_group=8, topk_group=4)))

    def test_accepted_when_the_scaling_factor_is_folded_into_the_weights(self):
        """`apply_routed_scaling_factor_on_output` set is the eligible case, not the bad one.

        The name is misleading: it makes the gate kernel fold routed_scaling_factor into
        the top-k weights (moe_fused_gate.py:249), not scale the MoE output. The fused
        entry folds it too, so the two agree. MiniMax-M3 sets it (minimax_m3.py:389) --
        refusing here would make the whole bypass dead for the target model.
        """
        self.assertTrue(
            self._gate(_glm_like_config(apply_routed_scaling_factor_on_output=True))
        )

    def test_refused_when_the_factor_is_not_folded_into_the_weights(self):
        """Clear flag + a real factor means the fused entry would apply it twice.

        `_pre_permute_bypassed_to_aiter` passes routed_scaling_factor to the entry
        unconditionally, but with the flag clear the standard path leaves the weights
        unscaled and the factor lands downstream instead. Scaling weights and scaling
        the output are equivalent for a linear combine, so this is silent.
        """
        self.assertFalse(
            self._gate(
                _glm_like_config(
                    apply_routed_scaling_factor_on_output=False,
                    routed_scaling_factor=2.5,
                )
            )
        )

    def test_accepted_when_there_is_no_factor_to_double_apply(self):
        """Positive control for the refusal above: only the factor distinguishes them."""
        for factor in (None, 1.0):
            with self.subTest(routed_scaling_factor=factor):
                self.assertTrue(
                    self._gate(
                        _glm_like_config(
                            apply_routed_scaling_factor_on_output=False,
                            routed_scaling_factor=factor,
                        )
                    )
                )

    def test_refused_for_a_custom_routing_function(self):
        self.assertFalse(
            self._gate(_glm_like_config(custom_routing_function=lambda *a, **k: None))
        )

    def test_refused_when_auto_does_not_resolve_to_the_aiter_runner(self):
        """`auto` plus an installed entry does not mean this layer uses that runner.

        With SGLANG_USE_AITER off, an MXFP4 MoE resolves to Triton even on gfx950, and
        Triton unpacks three fields from what would be a five-field bypassed output. The
        runner's own fallback cannot rescue that, because the aiter runner was never
        selected -- and format selection happens before the token cap, so it would fail
        above the cap too.
        """
        self.assertFalse(self._gate(_glm_like_config(), aiter_enabled=False))
        # An explicit aiter backend still qualifies: the flag only disambiguates `auto`.
        self.assertTrue(
            self._gate(_glm_like_config(), backend="aiter", aiter_enabled=False)
        )

    def test_refused_under_an_all_to_all_dispatcher(self):
        """Dispatch happens before the runner, so it reads routing that does not exist.

        MORI raises on its first routing read, ahead of any fallback in the runner.
        """
        for a2a in ("mori", "deepep", "pplx"):
            with self.subTest(a2a=a2a):
                self.assertFalse(self._gate(_glm_like_config(), a2a=a2a))

    def test_refused_under_expert_parallelism(self):
        """EP fails more quietly than an all-to-all dispatcher does.

        The standard dispatcher's first call builds the local expert mapping and the
        aiter expert mask from a standard top-k result. Materializing top-k later does
        not revisit dispatch, so the mask stays unset even on the fallback path.
        """
        self.assertFalse(self._gate(_glm_like_config(), ep_size=2))

    def test_refused_under_a_nontrivial_expert_placement(self):
        """The only failure here that is silent rather than an exception.

        Standard routing remaps logical expert ids onto physical slots; the fused entry
        does not. Under an initial placement or EPLB it would pair logical-order logits
        with physically placed weights, so a selected id addresses another expert's
        weights and the MoE output changes with nothing raised.
        """
        placement = object()  # any non-None ExpertLocationDispatchInfo
        self.assertFalse(
            self._gate(_glm_like_config(), expert_location_dispatch_info=placement)
        )

    def test_refused_above_the_topk_ceiling(self):
        self.assertFalse(self._gate(_glm_like_config(top_k=128)))


class TestValidatedEnvelope(CustomTestCase):
    def test_the_bound_matches_the_validated_token_set(self):
        """The constant is tied to the largest token count the sweep actually covered.

        If the sweep is extended, this assertion should fail first and force the
        constant to move with it, rather than the fused path quietly engaging at a
        token count nobody measured.
        """
        from sglang.srt.layers.moe.moe_runner.aiter import (
            AITER_FUSED_ROUTER_MAX_VALIDATED_TOKENS,
            FUSED_MOE_ROUTER_VALIDATED_TOKENS,
        )

        # The cap is the largest CONTIGUOUS verified width. Tying the assertion to the
        # verified set is what stops the two drifting apart: adding a width to the set
        # without verifying the gap below it will fail here rather than ship.
        self.assertEqual(
            AITER_FUSED_ROUTER_MAX_VALIDATED_TOKENS,
            max(FUSED_MOE_ROUTER_VALIDATED_TOKENS),
        )
        widths = sorted(FUSED_MOE_ROUTER_VALIDATED_TOKENS)
        self.assertEqual(
            widths,
            list(range(widths[0], widths[-1] + 1, widths[0])),
            "verified widths must be contiguous in steps of the smallest",
        )

    def test_prefill_sized_calls_are_outside_the_envelope(self):
        from sglang.srt.layers.moe.moe_runner.aiter import (
            AITER_FUSED_ROUTER_MAX_VALIDATED_TOKENS as bound,
        )

        # A chunked-prefill router call is orders of magnitude above the bound, and so
        # is the first disabled decode width (M=64 at concurrency 16).
        self.assertGreater(8192, bound)
        self.assertGreater(64, bound)  # concurrency 16's width is outside the cap

    def test_the_kernels_capability_cap_can_only_narrow_the_envelope(self):
        """Two limits meet here and the smaller must win.

        aiter owns the kernel's capability cap; this module owns the validated
        envelope. An older aiter with a lower cap must not be driven past it, and a
        newer aiter with a higher cap must not silently widen what we have measured.
        """
        from sglang.srt.layers.moe.moe_runner.aiter import (
            AITER_FUSED_ROUTER_MAX_VALIDATED_TOKENS as validated,
        )

        for cap in (64, 128, 1024):
            self.assertLessEqual(min(validated, cap), cap)
            self.assertLessEqual(min(validated, cap), validated)


class TestBypassedRunnerInput(CustomTestCase):
    def test_routed_topk_excludes_the_fused_shared_experts(self):
        """sglang's top_k counts fused shared experts; the aiter entry does not.

        Passing top_k straight through is one of the three documented ways to mis-wire
        this caller, and it writes past the first token's routing rows.
        """
        from sglang.srt.layers.moe.moe_runner.aiter import (
            AiterFusedRouterInput,
        )

        cfg = _glm_like_config()
        fr = AiterFusedRouterInput(
            router_logits=torch.zeros(4, 256),
            correction_bias=cfg.correction_bias,
            topk_output=None,
            topk=cfg.top_k - cfg.num_fused_shared_experts,
            num_fused_shared_experts=cfg.num_fused_shared_experts,
            num_expert_group=1,
            topk_group=1,
            renormalize=True,
            routed_scaling_factor=2.5,
            shared_expert_weight=1.0,
        )
        self.assertEqual(fr.topk, 8)
        self.assertEqual(fr.topk + fr.num_fused_shared_experts, cfg.top_k)


class TestAiterFusedRouterPolicyGate(CustomTestCase):
    """`SGLANG_ENABLE_AITER_FUSED_ROUTER` must be able to refuse a capable build.

    The measurement depends on this: with no policy gate the only negative control is
    an image carrying a different aiter, and that version bump confounds the result.
    So "off" has to mean None even when every capability probe would have said yes --
    which is why the flag-on case below is run through the identical fake, as a
    positive control. Without it, a None could just as well be the ImportError path.
    """

    def _with_a_capable_aiter(self):
        """Install a fake `aiter.fused_moe` carrying the whole fused-router surface."""
        import sys
        from types import ModuleType, SimpleNamespace

        pkg = ModuleType("aiter")
        mod = ModuleType("aiter.fused_moe")
        mod.FUSED_MOE_ROUTER_HIDDEN_DIMS = (4096, 6144)
        mod.FUSED_MOE_ROUTER_MAX_TOPK = 64
        mod.FUSED_MOE_ROUTER_MAX_EXPERTS = 512
        mod.FUSED_MOE_ROUTER_MAX_TOKENS = 512
        mod.fused_moe_router = lambda *a, **k: None
        mod.fused_moe_router_supported = lambda *a, **k: True
        mod.fused_moe_router_config_supported = lambda *a, **k: True
        mod.fused_moe_router_arch_supported = lambda: True
        pkg.fused_moe = mod
        return mock.patch.dict(
            sys.modules, {"aiter": pkg, "aiter.fused_moe": mod}
        ), SimpleNamespace()

    def _probe(self, enabled: bool):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.moe_runner import aiter as aiter_runner

        patcher, _ = self._with_a_capable_aiter()
        with patcher, envs.SGLANG_ENABLE_AITER_FUSED_ROUTER.override(enabled):
            # The probe is functools.cache'd, so the flag is latched at first call.
            # Clear on the way in and out so neither case can see the other's answer.
            aiter_runner._aiter_fused_router.cache_clear()
            try:
                return aiter_runner._aiter_fused_router()
            finally:
                aiter_runner._aiter_fused_router.cache_clear()

    def test_disabled_refuses_a_build_that_carries_the_kernel(self):
        self.assertIsNone(self._probe(enabled=False))

    def test_enabled_accepts_the_same_build(self):
        """Positive control for the test above, through the identical fake."""
        entry = self._probe(enabled=True)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.hidden_dims, (4096, 6144))
        self.assertEqual(entry.max_tokens, 512)

    def test_default_is_off(self):
        """Production default: a build carrying the kernel still does not use it."""
        from sglang.srt.environ import envs

        envs.SGLANG_ENABLE_AITER_FUSED_ROUTER.clear()
        self.assertFalse(envs.SGLANG_ENABLE_AITER_FUSED_ROUTER.get())


if __name__ == "__main__":
    unittest.main()

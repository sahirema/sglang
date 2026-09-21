import types
import unittest
from unittest.mock import patch

from sglang.srt.models import minimax_m3, minimax_m3_vl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# The shipped amd/MiniMax-M3-MXFP4 checkpoint resolves to
# MiniMaxM3SparseForConditionalGeneration (top-level architectures), so the VL
# class -- not the text-only one -- is the gate the loader actually asks.
_GATES = {
    "text": minimax_m3.MiniMaxM3SparseForCausalLM,
    "vl": minimax_m3_vl.MiniMaxM3SparseForConditionalGeneration,
}
_MODULES = {"text": minimax_m3, "vl": minimax_m3_vl}


def _hf_config(kind, *, n_shared_experts=1):
    text = types.SimpleNamespace(n_shared_experts=n_shared_experts)
    return text if kind == "text" else types.SimpleNamespace(text_config=text)


class TestSharedExpertsFusionDisableReason(CustomTestCase):
    """The two MiniMax-M3 entry classes must gate shared-experts fusion alike.

    Each class carries its own copy of the platform checks, and only the VL one
    is reachable for the shipped checkpoint. A ROCm fix applied to one file and
    not the other leaves fusion silently off (or silently on) depending on which
    architecture string the config names -- no error, just a different model.
    """

    def _reason(self, kind, *, is_cuda, is_hip, gfx95, device_sm, ep_size=1, quant=None):
        mod = _MODULES[kind]
        with (
            patch.object(mod, "_is_cuda", is_cuda),
            patch.object(mod, "_is_hip", is_hip),
            patch.object(mod, "_is_gfx95_supported", gfx95),
            patch.object(mod, "_device_sm", device_sm),
            patch.object(
                mod,
                "get_parallel",
                lambda: types.SimpleNamespace(moe_ep_size=ep_size),
            ),
            patch.object(
                mod,
                "get_moe_a2a_backend",
                lambda: types.SimpleNamespace(is_deepep=lambda: False),
            ),
        ):
            return _GATES[kind].shared_experts_fusion_disable_reason(
                _hf_config(kind), quant
            )

    def _both(self, **kwargs):
        reasons = {k: self._reason(k, **kwargs) for k in _GATES}
        self.assertEqual(
            reasons["text"],
            reasons["vl"],
            f"gates disagree for {kwargs}: {reasons}",
        )
        return reasons["vl"]

    def test_gfx950_enables_fusion(self):
        # _device_sm is not None on ROCm; the SM80 check must not reject it.
        self.assertIsNone(
            self._both(is_cuda=False, is_hip=True, gfx95=True, device_sm=0)
        )

    def test_rocm_below_gfx95_is_rejected(self):
        self.assertIn(
            "gfx950",
            self._both(is_cuda=False, is_hip=True, gfx95=False, device_sm=0),
        )

    def test_cuda_still_enabled(self):
        self.assertIsNone(
            self._both(is_cuda=True, is_hip=False, gfx95=False, device_sm=90)
        )

    def test_cuda_below_sm80_is_rejected(self):
        self.assertIn(
            "SM80", self._both(is_cuda=True, is_hip=False, gfx95=False, device_sm=70)
        )

    def test_neither_platform_is_rejected(self):
        self.assertIn(
            "CUDA or ROCm",
            self._both(is_cuda=False, is_hip=False, gfx95=False, device_sm=None),
        )

    def test_no_shared_experts_in_config(self):
        for kind in _GATES:
            mod = _MODULES[kind]
            with (
                patch.object(mod, "_is_cuda", True),
                patch.object(mod, "_is_hip", False),
            ):
                self.assertIn(
                    "No shared experts",
                    _GATES[kind].shared_experts_fusion_disable_reason(
                        _hf_config(kind, n_shared_experts=0), None
                    ),
                )

    def test_expert_parallelism_still_rejected_on_rocm(self):
        # The ROCm branch must not bypass the checks that follow it.
        self.assertIn(
            "expert",
            self._both(
                is_cuda=False, is_hip=True, gfx95=True, device_sm=0, ep_size=2
            ),
        )

    def test_modelopt_mixed_still_rejected_on_rocm(self):
        quant = types.SimpleNamespace(get_name=lambda: "modelopt_mixed")
        self.assertIn(
            "quantization formats",
            self._both(
                is_cuda=False, is_hip=True, gfx95=True, device_sm=0, quant=quant
            ),
        )


if __name__ == "__main__":
    unittest.main()

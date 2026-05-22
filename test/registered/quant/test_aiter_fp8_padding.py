"""Unit tests for the GLM-4.6V FP8 padding helpers.

Covers:
- ``_get_or_alloc_padded_activation_scratch`` shape, dtype, tail-zero, and
  process-global sharing semantics.
- ``_prewarm_aiter_fp8_padded_scratch`` allocation behaviour for a fake
  module tree that mimics what compressed-tensors W8A8 FP8 produces.

Coverage of ``apply_fp8_ptpc_linear`` itself requires the ROCm AIter build
and is therefore exercised through the GLM-4.6V TP=2/4/8 smokes in
``docs/examples/glm4v`` rather than this CPU unit test.
"""

import unittest

import torch

try:
    from sglang.srt.layers.quantization.fp8_utils import (
        _AITER_FP8_ACTIVATION_PAD_SCRATCH,
        _get_or_alloc_padded_activation_scratch,
    )

    _FP8_UTILS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - defensive
    _FP8_UTILS_IMPORT_ERROR = exc

try:
    from sglang.srt.model_executor.cuda_graph_runner import (
        _prewarm_aiter_fp8_padded_scratch,
    )

    _CUDA_GRAPH_RUNNER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - defensive
    _CUDA_GRAPH_RUNNER_IMPORT_ERROR = exc


@unittest.skipIf(
    _FP8_UTILS_IMPORT_ERROR is not None,
    f"sglang fp8_utils not importable in this environment: {_FP8_UTILS_IMPORT_ERROR}",
)
class TestGetOrAllocPaddedActivationScratch(unittest.TestCase):
    def setUp(self) -> None:
        _AITER_FP8_ACTIVATION_PAD_SCRATCH.clear()

    def tearDown(self) -> None:
        _AITER_FP8_ACTIVATION_PAD_SCRATCH.clear()

    def test_shape_and_dtype(self):
        q = torch.zeros((16, 5472), dtype=torch.bfloat16)
        scratch = _get_or_alloc_padded_activation_scratch(q, padded_k=5632)
        self.assertEqual(scratch.shape, (16, 5632))
        self.assertEqual(scratch.dtype, torch.bfloat16)
        self.assertEqual(scratch.device, q.device)

    def test_tail_is_zero_at_alloc(self):
        q = torch.zeros((4, 1368), dtype=torch.bfloat16)
        scratch = _get_or_alloc_padded_activation_scratch(q, padded_k=1536)
        self.assertTrue(torch.all(scratch[:, 1368:] == 0))

    def test_idempotent_returns_same_tensor(self):
        q = torch.zeros((8, 2736), dtype=torch.bfloat16)
        a = _get_or_alloc_padded_activation_scratch(q, padded_k=2816)
        b = _get_or_alloc_padded_activation_scratch(q, padded_k=2816)
        self.assertIs(a, b)

    def test_two_layers_share_scratch(self):
        q1 = torch.zeros((32, 5472), dtype=torch.bfloat16)
        q2 = torch.zeros((32, 5472), dtype=torch.bfloat16)
        a = _get_or_alloc_padded_activation_scratch(q1, padded_k=5632)
        b = _get_or_alloc_padded_activation_scratch(q2, padded_k=5632)
        self.assertIs(a, b)

    def test_different_keys_get_distinct_scratches(self):
        q_small = torch.zeros((8, 1368), dtype=torch.bfloat16)
        q_big = torch.zeros((16, 1368), dtype=torch.bfloat16)
        a = _get_or_alloc_padded_activation_scratch(q_small, padded_k=1536)
        b = _get_or_alloc_padded_activation_scratch(q_big, padded_k=1536)
        self.assertIsNot(a, b)
        self.assertEqual(a.shape[0], 8)
        self.assertEqual(b.shape[0], 16)

    def test_preserves_tail_zero_after_head_overwrite(self):
        q = torch.randn((4, 5472), dtype=torch.bfloat16)
        scratch = _get_or_alloc_padded_activation_scratch(q, padded_k=5632)
        scratch[:, :5472].copy_(q)
        self.assertTrue(torch.equal(scratch[:, :5472], q))
        self.assertTrue(torch.all(scratch[:, 5472:] == 0))


class _FakeLinear(torch.nn.Module):
    """Mimics a compressed-tensors W8A8 FP8 layer with padding metadata."""

    def __init__(
        self,
        n: int,
        original_k: int,
        padded_k: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        is_shuffled: bool = True,
    ) -> None:
        super().__init__()
        # Allocate at padded width to mirror runtime ``layer.weight`` shape.
        weight = torch.zeros((n, padded_k), dtype=dtype, device=device)
        weight.aiter_original_k = original_k
        weight.aiter_padded_k = padded_k
        weight.aiter_k_padding = padded_k - original_k
        weight.aiter_is_shuffled = is_shuffled
        # Use register_buffer so module.modules()/getattr lookups behave like
        # the real Parameter case without invoking Parameter.__new__ (which
        # would strip the metadata).
        self.weight = weight  # type: ignore[assignment]


class _UnpaddedLinear(torch.nn.Module):
    def __init__(self, n: int = 16, k: int = 4096) -> None:
        super().__init__()
        self.weight = torch.zeros((n, k), dtype=torch.bfloat16)


@unittest.skipIf(
    _CUDA_GRAPH_RUNNER_IMPORT_ERROR is not None
    or _FP8_UTILS_IMPORT_ERROR is not None,
    "sglang cuda_graph_runner / fp8_utils not importable in this environment",
)
class TestPrewarmAiterFp8PaddedScratch(unittest.TestCase):
    def setUp(self) -> None:
        _AITER_FP8_ACTIVATION_PAD_SCRATCH.clear()

    def tearDown(self) -> None:
        _AITER_FP8_ACTIVATION_PAD_SCRATCH.clear()

    def test_no_op_on_unpadded_only_model(self):
        """The prewarm hook is a real-world ``apply_fp8_ptpc_linear``
        prerequisite. When no weight carries padding metadata, the cache
        stays empty.
        """
        model = torch.nn.Sequential(_UnpaddedLinear(), _UnpaddedLinear())
        # The helper short-circuits on non-HIP, so we patch the gating flag
        # locally rather than relying on the imported value.
        from sglang.srt.model_executor import cuda_graph_runner as cgr

        original = cgr._is_hip
        cgr._is_hip = True
        try:
            _prewarm_aiter_fp8_padded_scratch(
                model, capture_bs=[1, 4, 16], num_tokens_per_bs=1
            )
        finally:
            cgr._is_hip = original
        self.assertEqual(len(_AITER_FP8_ACTIVATION_PAD_SCRATCH), 0)

    def test_allocates_one_scratch_per_unique_m_padded_k(self):
        from sglang.srt.model_executor import cuda_graph_runner as cgr

        model = torch.nn.ModuleList(
            [
                _FakeLinear(n=4096, original_k=5472, padded_k=5632),
                _FakeLinear(n=4096, original_k=5472, padded_k=5632),
                _FakeLinear(n=10944, original_k=4096, padded_k=4096),
                _FakeLinear(n=4096, original_k=2736, padded_k=2816),
            ]
        )
        capture_bs = [1, 16, 64]
        original = cgr._is_hip
        cgr._is_hip = True
        try:
            _prewarm_aiter_fp8_padded_scratch(
                model, capture_bs=capture_bs, num_tokens_per_bs=1
            )
        finally:
            cgr._is_hip = original

        # Two unique (padded_k != original_k) layers (5472->5632 and
        # 2736->2816) times three batch sizes = 6 scratch tensors. The
        # 4096->4096 layer contributes nothing because padding is a no-op.
        self.assertEqual(len(_AITER_FP8_ACTIVATION_PAD_SCRATCH), 2 * 3)

        for (device, dtype, m, padded_k), tensor in (
            _AITER_FP8_ACTIVATION_PAD_SCRATCH.items()
        ):
            self.assertEqual(tensor.shape, (m, padded_k))
            self.assertEqual(tensor.dtype, dtype)
            self.assertEqual(tensor.device, device)

    def test_idempotent_second_call_does_not_grow_cache(self):
        from sglang.srt.model_executor import cuda_graph_runner as cgr

        model = torch.nn.ModuleList(
            [_FakeLinear(n=4096, original_k=1368, padded_k=1536)]
        )
        original = cgr._is_hip
        cgr._is_hip = True
        try:
            _prewarm_aiter_fp8_padded_scratch(
                model, capture_bs=[2, 4], num_tokens_per_bs=1
            )
            first = dict(_AITER_FP8_ACTIVATION_PAD_SCRATCH)
            _prewarm_aiter_fp8_padded_scratch(
                model, capture_bs=[2, 4], num_tokens_per_bs=1
            )
            second = _AITER_FP8_ACTIVATION_PAD_SCRATCH
        finally:
            cgr._is_hip = original

        self.assertEqual(set(first.keys()), set(second.keys()))
        for key, tensor in first.items():
            self.assertIs(tensor, second[key])

    def test_num_tokens_per_bs_scales_m(self):
        from sglang.srt.model_executor import cuda_graph_runner as cgr

        model = torch.nn.ModuleList(
            [_FakeLinear(n=4096, original_k=1368, padded_k=1536)]
        )
        original = cgr._is_hip
        cgr._is_hip = True
        try:
            _prewarm_aiter_fp8_padded_scratch(
                model, capture_bs=[2], num_tokens_per_bs=4
            )
        finally:
            cgr._is_hip = original

        # bs=2 * num_tokens_per_bs=4 -> M=8 should be the only entry.
        ms = {key[2] for key in _AITER_FP8_ACTIVATION_PAD_SCRATCH}
        self.assertEqual(ms, {8})


if __name__ == "__main__":
    unittest.main()

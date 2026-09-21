"""Equivalence tests for the fused decode ``req_to_token`` write.

``write_decode_req_to_token_triton`` replaces::

    req_to_token[(req_pool_indices, locs)] = out_cache_loc.to(torch.int32)

with one Triton launch. The contract it has to honour is not "writes the right
values" alone but "is indistinguishable from ``index_put_``", so every case
below compares against that expression evaluated on a cloned table rather than
against hand-written expected values.

The padding-row case is the one worth stating outright: cuda-graph padded
batches carry ``req_pool_indices == 0`` and rely on row 0 absorbing the dummy
write. A kernel that masked those lanes out would pass a naive test and diverge
only on padded batches. ``test_padding_rows_are_written`` pins that down.

Not covered: index arithmetic beyond 2**31, which would need a >8 GB table to
reach. The kernel widens to int64 before multiplying by the row stride; that is
an inspection-level guarantee here, not a tested one.

Skipped on CPU -- Triton needs a GPU.

    python -m pytest test/registered/unit/mem_cache/test_decode_req_to_token_write.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

_HAS_CUDA = torch.cuda.is_available()

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

MAX_CONTEXT_LEN = 96
NUM_ROWS = 17  # pool size + 1 padding row, as ReqToTokenPool allocates


def _fresh_table(device):
    """A table pre-filled with a recognisable pattern, so an unwritten cell and
    a cell written with zero are distinguishable."""
    return (
        torch.arange(
            NUM_ROWS * MAX_CONTEXT_LEN, dtype=torch.int32, device=device
        ).reshape(NUM_ROWS, MAX_CONTEXT_LEN)
        + 1000
    )


def _reference(table, req_pool_indices, locs, out_cache_loc):
    ref = table.clone()
    ref[(req_pool_indices, locs)] = out_cache_loc.to(torch.int32)
    return ref


@unittest.skipUnless(_HAS_CUDA, "Triton decode write kernel requires a GPU")
class TestDecodeReqToTokenWrite(unittest.TestCase):
    def setUp(self):
        from sglang.kernels.ops.memory.common import (
            write_decode_req_to_token_triton,
        )

        self.fused = write_decode_req_to_token_triton
        self.device = torch.device("cuda")

    def _assert_matches_index_put(self, req_pool_indices, locs, out_cache_loc):
        table = _fresh_table(self.device)
        expected = _reference(table, req_pool_indices, locs, out_cache_loc)

        got = table.clone()
        self.fused(got, req_pool_indices, locs, out_cache_loc)

        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        return got

    def test_matches_index_put_typical_decode(self):
        bs = 8
        req_pool_indices = torch.arange(1, bs + 1, device=self.device)
        locs = torch.arange(5, 5 + bs, device=self.device)
        out_cache_loc = torch.arange(300, 300 + bs, device=self.device)
        self._assert_matches_index_put(req_pool_indices, locs, out_cache_loc)

    def test_padding_rows_are_written(self):
        """Padded cuda-graph lanes (req_pool_index 0) must land on row 0, not
        be skipped. This is the case a lane-masking kernel would get wrong."""
        bs = 6
        # Three live requests followed by three padding lanes, which is the
        # shape a padded decode batch takes.
        req_pool_indices = torch.tensor(
            [1, 2, 3, 0, 0, 0], dtype=torch.int64, device=self.device
        )
        locs = torch.tensor([7, 8, 9, 0, 0, 0], dtype=torch.int64, device=self.device)
        out_cache_loc = torch.tensor(
            [11, 12, 13, 14, 15, 16], dtype=torch.int64, device=self.device
        )

        got = self._assert_matches_index_put(req_pool_indices, locs, out_cache_loc)

        # Explicit, beyond the equivalence check: row 0 was actually touched.
        # index_put_ resolves the duplicate (0, 0) lanes nondeterministically,
        # so assert membership rather than a specific survivor.
        self.assertIn(int(got[0, 0]), (14, 15, 16))

    def test_narrowing_from_int64_is_value_preserving(self):
        bs = 4
        req_pool_indices = torch.arange(1, bs + 1, device=self.device)
        locs = torch.arange(bs, device=self.device)
        # Values near the int32 ceiling: a wrong-width store would corrupt them.
        out_cache_loc = torch.tensor(
            [0, 1, 2**31 - 2, 2**31 - 1], dtype=torch.int64, device=self.device
        )
        got = self._assert_matches_index_put(req_pool_indices, locs, out_cache_loc)
        self.assertEqual(int(got[3, 2]), 2**31 - 2)

    def test_untouched_cells_are_unchanged(self):
        bs = 3
        req_pool_indices = torch.tensor(
            [2, 4, 6], dtype=torch.int64, device=self.device
        )
        locs = torch.tensor([10, 11, 12], dtype=torch.int64, device=self.device)
        out_cache_loc = torch.tensor(
            [77, 88, 99], dtype=torch.int64, device=self.device
        )

        table = _fresh_table(self.device)
        before = table.clone()
        self.fused(table, req_pool_indices, locs, out_cache_loc)

        touched = torch.zeros_like(table, dtype=torch.bool)
        touched[(req_pool_indices, locs)] = True
        torch.testing.assert_close(
            table[~touched], before[~touched], rtol=0, atol=0
        )

    def test_batch_sizes_spanning_the_block(self):
        """BLOCK_SIZE is 256; cover below, at, and above one block.

        Targets are drawn as a permutation of flat table positions so the
        (row, loc) pairs are unique by construction -- duplicates resolve
        nondeterministically in both paths and would make the comparison
        flaky rather than wrong.
        """
        for bs in (1, 2, 255, 256, 257):
            with self.subTest(bs=bs):
                flat = torch.randperm(
                    NUM_ROWS * MAX_CONTEXT_LEN, device=self.device
                )[:bs]
                req_pool_indices = flat // MAX_CONTEXT_LEN
                locs = flat % MAX_CONTEXT_LEN
                out_cache_loc = torch.arange(
                    bs, dtype=torch.int64, device=self.device
                )
                self._assert_matches_index_put(
                    req_pool_indices, locs, out_cache_loc
                )

    def test_mismatched_shapes_raise(self):
        bs = 4
        req_pool_indices = torch.arange(1, bs + 1, device=self.device)
        locs = torch.arange(bs, device=self.device)
        out_cache_loc = torch.arange(bs * 2, device=self.device)
        table = _fresh_table(self.device)
        with self.assertRaises(AssertionError):
            self.fused(table, req_pool_indices, locs, out_cache_loc)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_tensors,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # write prefix
    num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
            value,
            mask=mask,
        )

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


@triton.jit
def _get_last_loc_safe_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result_i32,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
    PREFIX_DTYPE_IS_I64: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    if PREFIX_DTYPE_IS_I64:
        prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
        req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)
        token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    else:
        prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
        req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)
        token_index = req_pool_indices.to(tl.int64) * req_to_token_stride + (
            prefix_lens.to(tl.int64) - 1
        )

    token_mask = mask & (prefix_lens > 0)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)
    # Result stays int32 (req_to_token dtype); caller promotes after return.
    tl.store(result_i32 + offset, tokens, mask=mask)


def get_last_loc_triton_safe_i32(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    """`last_loc` in a SINGLE Triton launch, left in req_to_token's own int32
    dtype with no trailing promotion kernel.

    This is the no-promotion core of `get_last_loc_triton_safe`. Its result is
    bit-identical to what plain advanced indexing
    ``req_to_token[req_pool_indices, prefix_lens - 1]`` returns (that
    expression also yields req_to_token's int32 dtype), except that rows with
    ``prefix_lens == 0`` yield -1 here instead of wrapping around to the last
    column -- strictly the safer of the two.

    Prefer this over the torch expression on per-decode-step paths: the torch
    form costs two launches (an elementwise `sub` on a [bs] tensor, then an
    advanced-index gather) and decode on tiny [bs] tensors is bound by launch
    count, not arithmetic. Use `get_last_loc_triton_safe` instead when the
    caller needs the result in the index dtype rather than int32.
    """
    num_tokens = prefix_lens_tensor.shape[0]
    BLOCK_SIZE = 256
    result_i32 = torch.empty(
        num_tokens, dtype=torch.int32, device=prefix_lens_tensor.device
    )
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)
    _get_last_loc_safe_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result_i32,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        PREFIX_DTYPE_IS_I64=(prefix_lens_tensor.dtype == torch.int64),
    )
    return result_i32


def get_last_loc_triton_safe(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    """Fused `last_loc` Triton kernel whose in-kernel result buffer is int32
    (the dtype of req_to_token). The consumer-dtype promotion happens in
    torch after the kernel returns, so Triton never issues a mixed-width
    store -- avoiding the HIP int32->int64 store bug hit by the legacy kernel.
    """
    # `.to()` is a no-op returning self when the dtypes already match, so this
    # only costs a launch for callers that genuinely need a wider index dtype.
    return get_last_loc_triton_safe_i32(
        req_to_token, req_pool_indices_tensor, prefix_lens_tensor
    ).to(prefix_lens_tensor.dtype)


@triton.jit
def get_last_loc_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
    req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)

    token_mask = prefix_lens > 0
    token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)

    tl.store(result + offset, tokens, mask=mask)


def get_last_loc_triton(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    BLOCK_SIZE = 256
    num_tokens = prefix_lens_tensor.shape[0]
    result = torch.empty_like(prefix_lens_tensor)
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    get_last_loc_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE,
    )
    return result


@triton.jit
def _write_decode_req_to_token_kernel(
    req_to_token_ptr,  # [max_batch, max_context_len], int32
    req_pool_indices_ptr,  # [bs]
    locs_ptr,  # [bs]
    out_cache_loc_ptr,  # [bs]
    bs,
    req_to_token_ptr_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < bs

    req_pool_index = tl.load(req_pool_indices_ptr + offset, mask=mask, other=0)
    loc = tl.load(locs_ptr + offset, mask=mask, other=0)
    value = tl.load(out_cache_loc_ptr + offset, mask=mask, other=0)

    # Index arithmetic in int64: max_batch * max_context_len is within int32
    # for today's pools but not by a comfortable margin, and the row stride is
    # attacker-independent config, so widening here costs nothing measurable.
    token_index = req_pool_index.to(tl.int64) * req_to_token_ptr_stride + loc.to(
        tl.int64
    )
    # The cast is the point: narrowing here is what replaces the caller's
    # separate `out_cache_loc.to(torch.int32)` launch. Storing an explicitly
    # narrowed value keeps the store same-width rather than relying on Triton's
    # implicit conversion to the pointer dtype -- the same direction as the
    # extend path's store above, but stated rather than inferred.
    tl.store(req_to_token_ptr + token_index, value.to(tl.int32), mask=mask)


def write_decode_req_to_token_triton(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    locs: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> None:
    """One-launch equivalent of the decode `req_to_token` bookkeeping write.

    Replaces::

        req_to_token[(req_pool_indices, locs)] = out_cache_loc.to(torch.int32)

    which costs two launches -- an elementwise cast producing a throwaway [bs]
    int32 tensor, then an `index_put_` -- with a single fused cast-and-scatter.
    Decode issues this once per step on bs-sized tensors, where launch count
    dominates the arithmetic entirely.

    Semantics are those of `index_put_`, deliberately including its treatment of
    padding rows: every lane is written exactly as the torch expression would
    write it, with no filtering. Batches padded for cuda-graph replay carry
    `req_pool_indices == 0` and rely on row 0 of `req_to_token` absorbing the
    dummy write (see `ReqToTokenPool.__init__`, which allocates `size + 1` rows
    for exactly this reason), so a kernel that masked those lanes out instead
    would diverge from the torch path only on padded batches -- i.e. rarely, and
    not deterministically. It must not.

    Duplicate `(req_pool_index, loc)` pairs are not expected (each request owns
    a distinct row) and, as with `index_put_`, resolve nondeterministically.
    """
    bs = req_pool_indices.shape[0]
    # A mismatch here would silently write only the first `bs` values and
    # corrupt the KV index table rather than raise, so it is worth a check that
    # costs no launch and no device sync.
    assert locs.shape[0] == bs and out_cache_loc.shape[0] == bs, (
        f"decode req_to_token write expects [bs]-shaped operands, got "
        f"req_pool_indices={req_pool_indices.shape}, locs={locs.shape}, "
        f"out_cache_loc={out_cache_loc.shape}"
    )

    BLOCK_SIZE = 256
    grid = (triton.cdiv(bs, BLOCK_SIZE),)
    _write_decode_req_to_token_kernel[grid](
        req_to_token,
        req_pool_indices,
        locs,
        out_cache_loc,
        bs,
        req_to_token.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

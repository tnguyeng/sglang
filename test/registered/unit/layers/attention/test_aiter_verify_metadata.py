"""AITER backend on CPU: decode workspace, unified-verify metadata, DCP verify shard."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention import aiter_backend
from sglang.srt.layers.dcp import layout as dcp_layout
from sglang.srt.layers.dcp import planner as dcp_planner
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_BATCH_SIZE = 2
_VERIFY_TOKENS_PER_REQ = 17
_SEQ_LENS = torch.tensor([32, 64], dtype=torch.int32)


class _NoopKernel:
    def __getitem__(self, _grid):
        return lambda *_args, **_kwargs: None


def _fail_on_host_sync(_tensor):
    raise AssertionError("tensor.item() forces a device-to-host sync")


def _metadata(backend):
    return {
        name: value.tolist() if isinstance(value, torch.Tensor) else value
        for name, value in vars(backend.forward_metadata).items()
    }


def _verify_batch():
    return SimpleNamespace(
        batch_size=_BATCH_SIZE,
        input_ids=torch.zeros(_BATCH_SIZE * _VERIFY_TOKENS_PER_REQ),
        req_pool_indices=torch.arange(_BATCH_SIZE, dtype=torch.int32),
        seq_lens=_SEQ_LENS.clone(),
        seq_lens_cpu=_SEQ_LENS.clone(),
        seq_lens_sum=int(_SEQ_LENS.sum()),
        encoder_lens=None,
        spec_info=SimpleNamespace(draft_token_num=_VERIFY_TOKENS_PER_REQ - 1),
        forward_mode=ForwardMode.TARGET_VERIFY,
        out_cache_loc=None,
    )


def _unified_verify_backend():
    backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
    backend.device = torch.device("cpu")
    backend.use_mla = False
    backend._use_unified_verify = True
    backend.use_sliding_window_kv_pool = False
    backend.use_muse_attention_paths = False
    backend.muse_reuse_verify_lengths = False
    backend.page_size = 1
    backend.max_kv_len = 128
    backend.max_num_pages = 128
    backend.num_draft_tokens = _VERIFY_TOKENS_PER_REQ
    backend.qo_indptr = torch.zeros(_BATCH_SIZE + 1, dtype=torch.int32)
    backend.kv_indptr = torch.zeros(_BATCH_SIZE + 1, dtype=torch.int32)
    backend.cuda_graph_kv_indices = torch.zeros(128, dtype=torch.int32)
    backend.cuda_graph_kv_last_page_len = torch.zeros(_BATCH_SIZE, dtype=torch.int32)
    backend.req_to_token = torch.zeros((_BATCH_SIZE, 128), dtype=torch.int32)
    backend.cuda_graph_page_table = torch.zeros(
        (_BATCH_SIZE, backend.max_kv_len), dtype=torch.int32
    )

    def build_verify_unified_metadata(
        bs,
        _seq_lens,
        _req_pool_indices,
        draft_num,
        page_table_dest=None,
        swa_page_table_dest=None,
    ):
        page_table = (
            page_table_dest
            if page_table_dest is not None
            else torch.zeros((bs, 8), dtype=torch.int32)
        )
        qo_indptr = torch.arange(0, (bs + 1) * draft_num, draft_num, dtype=torch.int32)
        return page_table, qo_indptr, draft_num, swa_page_table_dest

    backend._build_verify_unified_metadata = build_verify_unified_metadata
    return backend


class TestAiterUnifiedVerifyWithoutHostSeqLens(CustomTestCase):
    """A backend that opts out of the host seq_lens mirror (needs_cpu_seq_lens=False)
    receives seq_lens_cpu=None on verify batches; unified verify sizes its metadata
    from the page table, so the metadata must match and no .item() may run."""

    def setUp(self):
        for target, replacement in (
            ("reset_verify_attn_plan_cache", lambda: None),
            ("create_flashinfer_kv_indices_triton", _NoopKernel()),
        ):
            patcher = patch.object(aiter_backend, target, replacement, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _assert_same_metadata_without_host_seq_lens(self, backend, init):
        forward_batch = _verify_batch()
        init(forward_batch)
        expected = _metadata(backend)
        self.assertEqual(backend.forward_metadata.max_q_len, _VERIFY_TOKENS_PER_REQ)

        forward_batch.seq_lens_cpu = None
        forward_batch.seq_lens_sum = None
        with patch.object(torch.Tensor, "item", _fail_on_host_sync):
            init(forward_batch)
        self.assertEqual(_metadata(backend), expected)

    def test_eager_init_skips_host_max(self):
        backend = _unified_verify_backend()
        self._assert_same_metadata_without_host_seq_lens(
            backend, backend.init_forward_metadata
        )

    def test_graph_replay_skips_device_max(self):
        backend = _unified_verify_backend()
        self._assert_same_metadata_without_host_seq_lens(
            backend,
            lambda fb: backend.init_forward_metadata_out_graph(fb, in_capture=False),
        )


_DCP_SIZE = 8
_DRAFT_TOKENS = 8
_DCP_MAX_CONTEXT_LEN = 128
# The second request fills the context, so its asm shard meets the static graph bound.
_DCP_SEQ_LENS = torch.tensor([32, _DCP_MAX_CONTEXT_LEN], dtype=torch.int32)


def _rank0_shard_lens(kv_lens):
    # Tokens are dealt round-robin from rank 0, so rank 0 holds ceil(len / DCP).
    return ((kv_lens + _DCP_SIZE - 1) // _DCP_SIZE).tolist()


def _dcp_verify_backend(asm_cprr):
    """An MLA backend at DCP8 rank 0, with the cuda graph buffers it allocates."""
    backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
    backend.device = torch.device("cpu")
    backend.use_mla = True
    backend.use_mla_dcp_asm = False
    backend._use_unified_verify = False
    backend.use_sliding_window_kv_pool = False
    backend.use_muse_attention_paths = False
    backend.muse_reuse_verify_lengths = False
    backend.decode_uses_page_table = False
    backend.skip_prefill = True
    backend.page_size = 1
    backend.dcp_world_size = _DCP_SIZE
    backend.num_draft_tokens = _DRAFT_TOKENS
    backend.cprr_kernel_heads = 96 if asm_cprr else 0
    backend.max_kv_len = _DCP_MAX_CONTEXT_LEN
    backend.max_num_pages = _DCP_MAX_CONTEXT_LEN
    backend.max_split_per_batch = 32
    backend.cprr_fast_mode = True
    backend.cprr_intra_batch_mode = False
    backend.qo_indptr = torch.zeros(_BATCH_SIZE + 1, dtype=torch.int32)
    backend.kv_indptr = torch.zeros(_BATCH_SIZE + 1, dtype=torch.int32)
    backend.req_to_token = torch.zeros(
        (_BATCH_SIZE, _DCP_MAX_CONTEXT_LEN + _DRAFT_TOKENS), dtype=torch.int32
    )
    backend.runner_ids = SimpleNamespace(
        full_v2p_table=None, needs_read_translate=False
    )
    backend._asm_cprr_supports_verify_shape = lambda _q_len: asm_cprr
    # The persistent-kernel work plan is aiter's; these tests check the KV shard.
    backend.make_mla_decode_meta_data_buffer = lambda *_a, **_k: (torch.zeros(1),) * 6
    backend.make_mla_meta_data = lambda *_args, **_kwargs: None
    backend.init_cuda_graph_state(
        max_bs=_BATCH_SIZE, max_num_tokens=_BATCH_SIZE * _DRAFT_TOKENS
    )
    return backend


class TestAiterDcpVerifyGraphReplay(CustomTestCase):
    """Graph replay must plan each rank's DCP verify shard from the lengths eager
    plans: the asm cp verify reads the draft tokens from the cache, so its shard
    spans them, and that shard must still fit the static graph buffers."""

    def setUp(self):
        parallel = SimpleNamespace(
            dcp_enabled=True, dcp_size=_DCP_SIZE, dcp_rank=0, attn_dcp_rank=0
        )
        for module, target, replacement in (
            (aiter_backend, "create_flashinfer_kv_indices_triton", _NoopKernel()),
            (aiter_backend, "create_mla_kv_page_table_for_dcp", _NoopKernel()),
            (aiter_backend, "compact_dcp_verify_token_table_to_ragged", _NoopKernel()),
            (dcp_planner, "update_kv_lens_and_indices", _NoopKernel()),
            # The asm cp route runs only with the persistent MLA kernels.
            (aiter_backend, "_use_mla_ps_kernel", True),
            (aiter_backend, "get_parallel", lambda: parallel),
            (dcp_planner, "get_parallel", lambda: parallel),
            (dcp_layout, "get_parallel", lambda: parallel),
        ):
            patcher = patch.object(module, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _replay_shard_lens(self, asm_cprr):
        backend = _dcp_verify_backend(asm_cprr)
        seq_lens = _DCP_SEQ_LENS.clone()
        backend._apply_cuda_graph_metadata(
            bs=_BATCH_SIZE,
            req_pool_indices=torch.arange(_BATCH_SIZE, dtype=torch.int32),
            seq_lens=seq_lens,
            seq_lens_sum=int(_DCP_SEQ_LENS.sum()),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=None,
            seq_lens_cpu=None,
            verify_tokens_per_req=_DRAFT_TOKENS,
        )
        # The planner localizes its KV lengths in place; the batch's must survive.
        self.assertEqual(seq_lens.tolist(), _DCP_SEQ_LENS.tolist())

        metadata = backend.forward_metadata
        shard_lens = metadata.local_kv_lens.view(_BATCH_SIZE, _DRAFT_TOKENS)[:, 0]
        self.assertLessEqual(
            int(shard_lens.max()), metadata.verify_token_table.shape[1]
        )
        if asm_cprr:
            self.assertLessEqual(
                int(metadata.cprr_kv_indptr[-1]), metadata.cprr_kv_indices.numel()
            )
        return shard_lens.tolist()

    def test_asm_cp_verify_shard_spans_the_draft_tokens(self):
        self.assertEqual(
            self._replay_shard_lens(asm_cprr=True),
            _rank0_shard_lens(_DCP_SEQ_LENS + _DRAFT_TOKENS),
        )

    def test_gluon_verify_shard_stays_the_prefix(self):
        # Gluon attends the draft tokens from k_window, on one rank only.
        self.assertEqual(
            self._replay_shard_lens(asm_cprr=False), _rank0_shard_lens(_DCP_SEQ_LENS)
        )


class _MHAPool:
    start_layer = 0
    layer_num = 1
    requires_physical_write_loc = False

    def __init__(self, kv_cache_layout):
        self.kv_cache_layout = kv_cache_layout
        self.layer_router = PlainPoolRouter(self, "full")

    def get_value_buffer(self, _layer_id):
        return torch.empty(1, 1, 8, 128)


def _init_mha_backend(kv_cache_layout, speculative_algorithm=None):
    pool = _MHAPool(kv_cache_layout)
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        is_multimodal=False,
        num_attention_heads=32,
        head_dim=128,
        get_num_kv_heads=lambda _tp_size: 8,
        attention_arch=AttentionArch.MHA,
        context_len=4096,
    )
    model_runner = SimpleNamespace(
        model_config=model_config,
        device=torch.device("cpu"),
        kv_cache_dtype=torch.bfloat16,
        req_to_token_pool=SimpleNamespace(
            row_width=4096, req_to_token=None, size=_BATCH_SIZE
        ),
        runner_ids=None,
        token_to_kv_pool=pool,
        is_draft_worker=False,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
    )
    spec = SimpleNamespace(
        speculative_num_draft_tokens=None,
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_algorithm=speculative_algorithm,
    )
    parallel = SimpleNamespace(attn_tp_size=1, attn_dcp_size=1)
    exec_config = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=False)
    )
    with (
        patch.object(
            aiter_backend, "get_schedule", lambda: SimpleNamespace(page_size=1)
        ),
        patch.object(aiter_backend, "get_exec", lambda: exec_config),
        patch.object(aiter_backend, "get_spec", lambda: spec),
        patch.object(aiter_backend, "get_parallel", lambda: parallel),
    ):
        return aiter_backend.AiterAttnBackend(model_runner, skip_prefill=True)


class TestAiterDecodeWorkspace(CustomTestCase):
    """Only paged_attention_ragged reads workspace_buffer; a SHUFFLE 5D pool decodes
    through pa_decode_gluon, so it must not allocate it."""

    def test_allocated_only_for_paged_ragged_decode(self):
        allocated = {
            layout: "workspace_buffer" in vars(_init_mha_backend(layout))
            for layout in ("nhd", "vectorized_5d")
        }
        self.assertEqual(allocated, {"nhd": True, "vectorized_5d": False})


_MASK_LEN = 6


def _dense_verify_backend(dspark):
    """A dense (non-MLA) backend on the non-unified target-verify path, the only
    path that builds a custom_mask into the CUDA graph."""
    backend = _unified_verify_backend()
    backend._use_unified_verify = False
    backend._dspark_verify = dspark
    backend.max_kv_len = backend.max_context_len
    backend.mask_indptr = torch.zeros(_BATCH_SIZE + 1, dtype=torch.int64)
    backend.cuda_graph_custom_mask = torch.zeros(16, dtype=torch.uint8)
    return backend


class TestAiterDsparkGraphVerifyMask(CustomTestCase):
    """DSpark verify never builds a custom_mask, but CUDA-graph capture passes a
    dummy non-None one. The mask must be chosen from config, not from that tensor,
    so capture and replay both use no mask; other speculative methods keep copying
    their real mask into the graph buffer."""

    def setUp(self):
        for target, replacement in (
            ("reset_verify_attn_plan_cache", lambda: None),
            ("create_flashinfer_kv_indices_triton", _NoopKernel()),
        ):
            patcher = patch.object(aiter_backend, target, replacement, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _custom_mask(self, backend, custom_mask, in_capture):
        forward_batch = _verify_batch()
        forward_batch.spec_info = SimpleNamespace(
            draft_token_num=_VERIFY_TOKENS_PER_REQ - 1, custom_mask=custom_mask
        )
        backend.init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        return backend.forward_metadata.custom_mask

    def test_dspark_has_no_mask_at_capture_or_replay(self):
        backend = _dense_verify_backend(dspark=True)
        dummy_mask = torch.ones(_MASK_LEN, dtype=torch.uint8)

        self.assertIsNone(self._custom_mask(backend, dummy_mask, in_capture=True))
        self.assertIsNone(self._custom_mask(backend, None, in_capture=False))
        # The graph's mask buffer is never written, so replay cannot read it stale.
        self.assertFalse(backend.cuda_graph_custom_mask.any())

    def test_real_mask_is_copied_into_the_graph_buffer(self):
        backend = _dense_verify_backend(dspark=False)
        mask = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.uint8)

        custom_mask = self._custom_mask(backend, mask, in_capture=False)

        self.assertIs(custom_mask, backend.cuda_graph_custom_mask)
        self.assertEqual(custom_mask[:_MASK_LEN].tolist(), mask.tolist())
        self.assertFalse(custom_mask[_MASK_LEN:].any())

    def test_missing_mask_stays_none_without_dspark(self):
        backend = _dense_verify_backend(dspark=False)
        self.assertIsNone(self._custom_mask(backend, None, in_capture=False))

    def test_dspark_flag_follows_the_speculative_algorithm(self):
        flags = {
            algorithm: _init_mha_backend("nhd", algorithm)._dspark_verify
            for algorithm in ("DSPARK", "EAGLE", None)
        }
        self.assertEqual(flags, {"DSPARK": True, "EAGLE": False, None: False})


if __name__ == "__main__":
    unittest.main()

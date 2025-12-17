from typing import TYPE_CHECKING, Any, Optional, Union, cast

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.models.interfaces import supports_mrope
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import LazyLoader, cdiv
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    split_attn_metadata,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm_gaudi.v1.worker.hpu_input_batch import CachedRequestState
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
from vllm.v1.worker.gpu_model_runner import PerLayerAttnMetadata
from vllm.v1.worker.ubatch_splitting import ubatch_split
from vllm.v1.worker.ubatch_utils import UBatchSlices

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    xgr_torch_compile = LazyLoader(
        "xgr_torch_compile",
        globals(),
        "xgrammar.kernels.apply_token_bitmask_inplace_torch_compile",
    )

logger = init_logger(__name__)


class OmniHPUModelRunner(HPUModelRunner):
    def _init_mrope_positions(self, req_state: CachedRequestState):
        """Initialize M-RoPE positions for multimodal inputs.

        Extracts multimodal feature metadata (image grids, video grids,
        audio features) and computes M-RoPE positions for proper positional
        encoding of multimodal tokens.

        Args:
            req_state: Cached request state containing multimodal features

        Raises:
            AssertionError: If the model does not support M-RoPE
        """
        image_grid_thw = []
        video_grid_thw = []
        second_per_grid_ts = []
        audio_feature_lengths = []
        use_audio_in_video = False
        for mm_feature in req_state.mm_features:
            mm_item = mm_feature.data
            if mm_item is None:
                continue
            mm_input = mm_item.get_data()
            if (t := mm_input.get("image_grid_thw")) is not None:
                image_grid_thw.append(t.tolist())
            if (t := mm_input.get("video_grid_thw")) is not None:
                video_grid_thw.append(t.tolist())
            if (t := mm_input.get("second_per_grid_ts")) is not None:
                second_per_grid_ts.append(t)
            if (t := mm_input.get("audio_feature_lengths")) is not None:
                audio_feature_lengths.append(t)
            # Check for use_audio_in_video
            use_audio_in_video_value = mm_input.get("use_audio_in_video")
            if use_audio_in_video_value is not None:
                use_audio_in_video = bool(use_audio_in_video_value.item())

        if supports_mrope(self.model):
            req_state.mrope_positions, req_state.mrope_position_delta = self.model.get_mrope_input_positions(
                req_state.prompt_token_ids,
                hf_config=self.model_config.hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )
        else:
            req_state.mrope_positions, req_state.mrope_position_delta = MRotaryEmbedding.get_input_positions_tensor(
                req_state.prompt_token_ids,
                hf_config=self.model_config.hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )

    @torch.inference_mode()
    def extract_multimodal_outputs(self, hidden_states: Union[torch.Tensor, list[torch.Tensor]]) -> dict:
        if hasattr(self.model, "have_multimodal_outputs") and self.model.have_multimodal_outputs:
            text_hidden_states = hidden_states.text_hidden_states
            multimodal_outputs = hidden_states.multimodal_outputs

        elif isinstance(hidden_states, torch.Tensor):
            text_hidden_states = hidden_states
            multimodal_outputs = {}
        elif isinstance(hidden_states, list):
            text_hidden_states = hidden_states[0]
            multimodal_outputs = {}
        else:
            raise ValueError(f"Invalid hidden states type: {type(hidden_states)}")
        return text_hidden_states, multimodal_outputs

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
        """
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode in {
            CUDAGraphMode.NONE,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.FULL,
        }

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = num_tokens // 2
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = cdiv(num_tokens, max_query_len)
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)

        total_num_scheduled_tokens = int(num_scheduled_tokens.sum())

        ubatch_slices = None
        num_tokens_after_padding = None

        # We currently only microbatch if the number of tokens is
        # over a certain threshold.
        if self.parallel_config.enable_dbo and allow_microbatching:
            ubatch_slices, ubatch_num_tokens_after_padding = ubatch_split(
                num_scheduled_tokens,
                total_num_scheduled_tokens,
                total_num_scheduled_tokens,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
            )
            # Currently when DBO is enabled `ubatch_split` returns
            # the num_tokens_after_padding for a single ubatch, but we have 2
            # TODO(sage,lucas): this is cruft that should be addressed in the
            # padding refactor.
            if ubatch_num_tokens_after_padding is not None:
                num_tokens_after_padding = ubatch_num_tokens_after_padding * 2

        # If we failed to microbatch, currently need to resynchronize
        # TODO(lucas,sage): we should be able to avoid this second sync by
        #  refactoring `get_dp_padding_ubatch` and `get_dp_padding` into
        #  a single `coordinate_batch_across_dp` function.
        if num_tokens_after_padding is None:
            num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
            num_tokens_after_padding = num_tokens + num_pad
        else:
            num_tokens_across_dp = num_tokens_after_padding
            num_tokens_after_padding = int(num_tokens_after_padding[0].item())

        attn_metadata: Optional[PerLayerAttnMetadata] = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            attn_metadata = {}
            if ubatch_slices is not None:
                attn_metadata = [dict() for _ in range(len(ubatch_slices))]

            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens = max_query_len
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()

            for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
                    seq_lens=self.seq_lens.gpu[:num_reqs],
                    seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],  # noqa: E501
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=max_query_len,
                    max_seq_len=self.max_model_len,
                    block_table_tensor=self.input_batch.block_table[kv_cache_group_id].get_device_tensor(num_reqs),
                    slot_mapping=self.input_batch.block_table[kv_cache_group_id].slot_mapping.gpu[:num_tokens],
                    causal=True,
                )
                for attn_group in self.attn_groups[kv_cache_group_id]:
                    if ubatch_slices is not None:
                        common_attn_metadata_list = split_attn_metadata(ubatch_slices, common_attn_metadata)
                        for ubid, common_attn_metadata in enumerate(common_attn_metadata_list):
                            assert common_attn_metadata.max_query_len == 1
                            attn_metadata_i = attn_group.get_metadata_builder(
                                ubatch_id=ubid
                            ).build_for_cudagraph_capture(common_attn_metadata)
                            for layer_name in attn_group.layer_names:
                                assert type(attn_metadata) is list
                                attn_metadata[ubid][layer_name] = attn_metadata_i
                    else:
                        assert type(attn_metadata) is dict
                        attn_metadata_i = attn_group.get_metadata_builder().build_for_cudagraph_capture(
                            common_attn_metadata
                        )  # noqa: E501
                        for layer_name in attn_group.layer_names:
                            attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens, remove_lora):
            model_kwargs = self._init_model_kwargs(num_tokens)
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
                model_kwargs = self._init_model_kwargs(num_tokens)
            else:
                input_ids = self.input_ids.gpu[:num_tokens]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens]
            else:
                positions = self.positions.gpu[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=self.max_num_tokens,
                        dtype=self.model_config.dtype,
                        device=self.device,
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(num_tokens, None, False)

            # filter out the valid batch descriptor
            _cg_mode, batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    BatchDescriptor(
                        num_tokens=num_tokens_after_padding,
                        uniform_decode=uniform_decode,
                    )
                )
                if not is_profile
                else (CUDAGraphMode.NONE, None)
            )
            if cudagraph_runtime_mode is not None:
                # we allow forcing NONE when the dispatcher disagrees to support
                # warm ups for cudagraph capture
                assert cudagraph_runtime_mode == CUDAGraphMode.NONE or cudagraph_runtime_mode == _cg_mode, (
                    f"Cudagraph runtime mode mismatch at dummy_run. "
                    f"Expected {_cg_mode}, but got {cudagraph_runtime_mode}."
                )
            else:
                cudagraph_runtime_mode = _cg_mode

            if ubatch_slices is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_after_padding = ubatch_slices[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_after_padding

            with (
                self.maybe_randomize_inputs(input_ids),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_after_padding,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    ubatch_slices=ubatch_slices,
                ),
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                self.drafter.dummy_run(num_tokens)

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
        return hidden_states, hidden_states[logit_indices]

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens_np: np.ndarray,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        ubatch_slices: Optional[UBatchSlices] = None,
        num_tokens_after_padding: Optional[torch.Tensor] = None,
    ) -> tuple[
        int,
        int,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[IntermediateTensors],
        dict[str, Any],
        Optional[dict[str, dict]],
    ]:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if ubatch_slices:
            assert num_tokens_after_padding is not None
            num_input_tokens = int(num_tokens_after_padding[0].item() * 2)
            self.pad_out_ubatch_slice(ubatch_slices, num_input_tokens)
        elif ubatch_slices is None:
            num_input_tokens = self._get_num_input_tokens(num_scheduled_tokens)
            num_pad, num_tokens_after_padding = self.get_dp_padding(num_input_tokens)
            num_input_tokens += num_pad

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        per_req_additional_information: Optional[dict[str, dict]] = None
        if self.supports_mm_inputs and get_pp_group().is_first_rank and not self.model_config.is_encoder_decoder:
            # Build multimodal inputs and overlay prompt embeds; collect per-request info
            per_req_additional_information = self._build_mm_inputs_and_overlays(
                scheduler_output, num_scheduled_tokens, num_scheduled_tokens_np
            )

            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = {
                **self._init_model_kwargs(num_scheduled_tokens),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and get_pp_group().is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = self.is_token_ids.gpu[:num_scheduled_tokens].nonzero(as_tuple=False).squeeze(1)
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.get_input_embeddings(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs(num_input_tokens)
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs(num_input_tokens)
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if self.model_config.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            encoder_inputs = self._extract_encoder_inputs(scheduler_output)
            model_kwargs.update(encoder_inputs)

        return (
            num_scheduled_tokens,
            num_input_tokens,
            num_tokens_after_padding,
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            per_req_additional_information,
        )

    def _decode_and_store_request_payloads(self, scheduler_output: "SchedulerOutput") -> None:
        """Decode per-request prompt_embeds and additional_information for newly
        scheduled requests and store them to CPU in the request state.
        This version avoids hard dependency on payload classes by duck-typing."""
        try:
            new_reqs = getattr(scheduler_output, "scheduled_new_reqs", [])
            if not new_reqs:
                return
            for nr in new_reqs:
                req_id = getattr(nr, "req_id", None) or getattr(nr, "request_id", None)
                if req_id is None:
                    continue
                # prompt_embeds
                payload_pe = getattr(nr, "prompt_embeds", None)
                pe_cpu = None
                if payload_pe is not None:
                    if isinstance(payload_pe, torch.Tensor):
                        pe_cpu = payload_pe.detach().to("cpu").contiguous()
                    else:
                        # Try duck-typing a payload with data/shape/dtype
                        data = getattr(payload_pe, "data", None)
                        shape = getattr(payload_pe, "shape", None)
                        if data is not None and shape is not None:
                            dt = np.dtype(getattr(payload_pe, "dtype", "float32"))
                            arr = np.frombuffer(data, dtype=dt)
                            arr = arr.reshape(shape)
                            pe_cpu = torch.from_numpy(arr)
                if pe_cpu is not None and req_id in self.requests:
                    setattr(self.requests[req_id], "prompt_embeds_cpu", pe_cpu)
                # additional_information
                payload_info = getattr(nr, "additional_information", None)
                if payload_info is not None:
                    info_dict = {}
                    if isinstance(payload_info, dict):
                        info_dict = payload_info
                    else:
                        # Try duck-typing a payload with entries, each entry may have
                        # tensor_data/tensor_dtype/tensor_shape or list_data
                        entries = getattr(payload_info, "entries", None)
                        if isinstance(entries, dict):
                            for k, entry in entries.items():
                                tensor_data = getattr(entry, "tensor_data", None)
                                if tensor_data is not None:
                                    dt = np.dtype(getattr(entry, "tensor_dtype", "float32"))
                                    arr = np.frombuffer(tensor_data, dtype=dt)
                                    arr = arr.reshape(getattr(entry, "tensor_shape", ()))
                                    info_dict[k] = torch.from_numpy(arr)
                                else:
                                    info_dict[k] = getattr(entry, "list_data", None)
                    if info_dict and req_id in self.requests:
                        setattr(self.requests[req_id], "additional_information_cpu", info_dict)
        except Exception as e:
            logger.error(f"Error decoding prompt_embeds / additional_information: {e}")

    def _gather_runtime_additional_information(self) -> list[dict]:
        """Gather per-request additional_information stored in request state in batch order."""
        per_req_runtime_info = []
        for req_id in self.input_batch.req_ids:
            req_state = self.requests.get(req_id)
            info = getattr(req_state, "additional_information_cpu", None) if req_state is not None else None
            per_req_runtime_info.append(info if isinstance(info, dict) else {})
        return per_req_runtime_info

    def _compute_request_token_spans(self, num_scheduled_tokens_np) -> list[tuple[int, int]]:
        """Compute (start, end) token spans for each request within the flattened step sequence."""
        req_token_spans: list[tuple[int, int]] = []
        for req_index in range(len(self.input_batch.req_ids)):
            start_offset = int(self.query_start_loc.cpu[req_index])
            sched_tokens = int(num_scheduled_tokens_np[req_index])
            req_token_spans.append((start_offset, start_offset + sched_tokens))
        return req_token_spans

    def _build_model_kwargs_extra(
        self,
        per_req_additional_information: Optional[dict[str, dict]],
        num_scheduled_tokens_np,
    ) -> dict:
        """Build extra keyword arguments passed to the model for this step, including:
        - additional_information_by_req_id: per-request additional information provided by preprocess for this step
        - runtime_additional_information: per-request additional information stored in request state
        - request_ids: the request id list in batch order
        - request_token_spans: token spans per request within the flattened sequence
        """
        model_kwargs_extra: dict[str, object] = {}
        if per_req_additional_information:
            model_kwargs_extra["additional_information_by_req_id"] = per_req_additional_information
        try:
            model_kwargs_extra["runtime_additional_information"] = self._gather_runtime_additional_information()
            model_kwargs_extra["request_ids"] = self.input_batch.req_ids
            model_kwargs_extra["request_token_spans"] = self._compute_request_token_spans(num_scheduled_tokens_np)
        except Exception:
            pass
        return model_kwargs_extra

    def _process_additional_information_updates(self, multimodal_outputs: object) -> None:
        """Process model-provided per-request additional_information updates and merge into request state."""
        try:
            if not isinstance(multimodal_outputs, dict):
                return
            if (
                "additional_information_update" not in multimodal_outputs
                and "additional_information_update_by_req_id" not in multimodal_outputs
            ):
                return
            updates_list = multimodal_outputs.get("additional_information_update")
            if isinstance(updates_list, list):
                for idx, upd in enumerate(updates_list):
                    if not isinstance(upd, dict) or idx >= len(self.input_batch.req_ids):
                        continue
                    req_id = self.input_batch.req_ids[idx]
                    self._merge_additional_information_update(req_id, upd)
            updates_map = multimodal_outputs.get("additional_information_update_by_req_id")
            if isinstance(updates_map, dict):
                for req_id, upd in updates_map.items():
                    if not isinstance(upd, dict):
                        continue
                    if req_id not in self.requests:
                        continue
                    self._merge_additional_information_update(req_id, upd)
        except Exception as e:
            logger.error(
                f"Error merging for requests:{self.input_batch.req_ids} "
                f"additional information update: {e}, with the multimodal_outputs "
                f"as {multimodal_outputs}"
            )

    def _build_mm_inputs_and_overlays(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
    ) -> dict[str, dict]:
        """Run MM encoder, fill scheduled inputs_embeds, and overlay prompt embeds;
        collect per-request additional_information for the prefill portion."""
        # Run the multimodal encoder if any.
        self._execute_mm_encoder(scheduler_output)
        mm_embeds = self._gather_mm_embeddings(scheduler_output)
        # Always use embeddings as inputs for MM models
        inputs_embeds_scheduled = self.model.get_input_embeddings(
            input_ids=self.input_ids.gpu[:num_scheduled_tokens],
            multimodal_embeddings=mm_embeds or None,
        )
        self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)
        # Reset per-step additional information collector (deprecated concat path)
        if hasattr(self, "_forward_additional_information"):
            self._forward_additional_information = None
        # Overlay and collect per-request info
        return self._collect_additional_information_for_prefill(num_scheduled_tokens_np)

    def _collect_additional_information_for_prefill(
        self,
        num_scheduled_tokens_np: np.ndarray,
    ) -> dict[str, dict]:
        """Overlay per-request prompt_embeds for the prefill portion and collect
        additional_information slices for this step. Returns a map req_id -> dict."""
        per_req_additional_information: dict[str, dict] = {}
        for req_index, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            pe_cpu = getattr(req_state, "prompt_embeds_cpu", None)
            addi_cpu = getattr(req_state, "additional_information_cpu", None)
            num_computed_tokens = int(self.input_batch.num_computed_tokens_cpu[req_index])
            prompt_len = len(req_state.prompt_token_ids)
            prompt_remaining = max(0, prompt_len - num_computed_tokens)
            sched_tokens = int(num_scheduled_tokens_np[req_index])
            overlay_len = min(sched_tokens, prompt_remaining)
            if overlay_len <= 0 and not (isinstance(addi_cpu, dict) and addi_cpu):
                continue
            if overlay_len > 0 and pe_cpu is not None:
                src = pe_cpu[num_computed_tokens : num_computed_tokens + overlay_len].to(
                    dtype=self.dtype, device=self.device, non_blocking=True
                )
                start_offset = int(self.query_start_loc.cpu[req_index])
                self.inputs_embeds[start_offset : start_offset + overlay_len].copy_(src)
            if addi_cpu is not None and isinstance(addi_cpu, dict):
                req_info: dict[str, object] = {}
                for k, v in addi_cpu.items():
                    if isinstance(v, torch.Tensor):
                        req_info[k] = v.detach().to("cpu").contiguous()
                    elif isinstance(v, list):
                        req_info[k] = v
                    else:
                        req_info[k] = v
                if req_info:
                    per_req_additional_information[req_id] = req_info
        return per_req_additional_information

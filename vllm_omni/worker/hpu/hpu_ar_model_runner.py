"""AR HPU Model Runner for vLLM-Omni.

Exposes per-request hidden representations via ModelRunnerOutput.pooler_output
and also outputs sampled tokens.
"""

from __future__ import annotations

import habana_frameworks.torch as htorch
import torch
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.multimodal.inputs import (BatchedTensorInputs, MultiModalKwargs, MultiModalKwargsItem)
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors, DraftTokenIds, ModelRunnerOutput,
                             AsyncModelRunnerOutput, KVConnectorOutput)
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm_gaudi.extension.ops import LoraMask
from vllm_gaudi.v1.worker.hpu_model_runner import (
    AsyncHPUModelRunnerOutput,
    ensure_decodes_first,
    get_kv_transfer_group,
    has_kv_transfer_group,
    set_forward_context,
    shallow_tuple,
    trim_attn_metadata,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.hpu.hpu_model_runner import OmniHPUModelRunner

logger = init_logger(__name__)


class HPUARModelRunner(OmniHPUModelRunner):
    """Autoregressive HPU model runner that returns hidden states per request.

    This runner follows the same preparation and forward path as HPUModelRunner
    (inputs assembly, multi-modal handling, TP/PP/DP integration, CUDA graphs),
    and additionally performs lightweight sampling so that sampled tokens are
    available in outputs. Hidden representations are taken at the same indices
    that HPUModelRunner would use for sampling/logits (i.e. `logits_indices`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _execute_model_generic_omni(self,
                                    token_ids,
                                    position_ids,
                                    attn_metadata,
                                    logits_indices,
                                    kv_caches,
                                    lora_logits_mask,
                                    lora_mask,
                                    warmup_mode=False,
                                    inputs_embeds=None,
                                    model_mm_kwargs=None):
        # FORWARD.
        batch_size = token_ids.size(0)
        seq_len = self._seq_len(attn_metadata)
        num_blocks = self._num_blocks(attn_metadata)
        if not self.unified_attn:
            self._check_config(batch_size, seq_len, num_blocks, attn_metadata, warmup_mode)
        else:
            self._check_unified_config(attn_metadata, logits_indices, warmup_mode)
        additional_kwargs = {}
        if htorch.utils.internal.is_lazy():
            use_graphs = self._use_graphs()
            additional_kwargs.update({"bypass_hpu_graphs": not use_graphs})
        else:
            # no hpu graphs for t.compile?
            use_graphs = False
        if self.model_has_chunked_attention:
            additional_kwargs.update({"model_has_chunked_attention": True})
        trimmed_attn_metadata = attn_metadata if self.unified_attn else trim_attn_metadata(attn_metadata)
        if self.is_driver_worker:
            model_event_name = ("model_forward_"
                                f"bs{batch_size}_"
                                f"seq{seq_len}_"
                                f"ctx{num_blocks}_"
                                f"graphs{'T' if use_graphs else 'F'}")
        else:
            model_event_name = 'model_executable'
        with self.profiler.record_event('internal', model_event_name):
            model_output = self.model.forward(input_ids=token_ids,
                                               positions=position_ids,
                                               attn_metadata=trimmed_attn_metadata,
                                               kv_caches=kv_caches,
                                               inputs_embeds=inputs_embeds,
                                               model_mm_kwargs=model_mm_kwargs,
                                               lora_mask=lora_mask,
                                               **additional_kwargs)
        # Omni specific
        # TODO(czhu15): handle the multimodal_outputs
        multimodal_outputs = model_output.multimodal_outputs
        hidden_states = model_output.text_hidden_states

        # NOTE(kzawora): returning hidden_states is required in prompt logprobs
        # scenarios, as they will do logit processing on their own
        if self.use_aux_hidden_state_outputs:
            non_flattened_hidden_states, aux_hidden_states = hidden_states
            hidden_states = non_flattened_hidden_states
        else:
            non_flattened_hidden_states = hidden_states
            aux_hidden_states = None

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = hidden_states[logits_indices]
        LoraMask.setLoraMask(lora_logits_mask)
        with self.profiler.record_event('internal', ('compute_logits'
                                                     f'{batch_size}_'
                                                     f'seq{seq_len}_ctx'
                                                     f'{num_blocks}')):
            logits = self.model.compute_logits(hidden_states)
        return non_flattened_hidden_states, aux_hidden_states, \
            hidden_states, logits

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        warmup_mode: bool = False,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        # based on vllm_gaudi.v1.worker.hpu_model_runner.HPUModelRunner.execute_model
        if self.unified_attn:
            return self.unified_execute_model(scheduler_output, warmup_mode)
        # NOTE(kzawora): Since scheduler doesn't differentiate between prefills
        # and decodes, we must handle mixed batches. In _update_states we make
        # sure that first self.input_batch.num_decodes requests are decodes,
        # and remaining ones until the end are prefills. _update_states also
        # handles changes in request cache based on scheduler outputs and
        # previous iterations (e.g. keeping block tables and context lengths up
        # to date, creating, pruning and updating request caches,
        # and some more stuff)

        # If num_decodes == self.input_batch.num_reqs, then batch is all decode, and only a single decode forward pass will be executed in this method. # noqa
        # If num_decodes == 0, then batch is all prefill, and only prefill forward passes will be executed  in this method. # noqa
        # If neither apply, then batch is mixed, and both prefill and decode forward passes will be executed in this method. # noqa

        # First, we will execute all decodes (if any) in a single batch,
        # then we'll execute prefills in batches of up to max_prefill_batch_size elements. # noqa
        # All shapes used in forward passes are bucketed appropriately to mitigate risk of graph recompilations. # noqa

        # We perform sampling directly after executing each forward pass
        # Everything is done asynchronously - the only sync point is the place
        # where we copy the generated tokens back to the host.

        # Example: If a batch has 6 requests, 3 prefills and 3 decodes, the unprocessed sequences in batch will be laid as follows: # noqa
        # [D0, D1, D2, P0, P1, P2]
        # If we assume max_prefill_batch_size=2, the flow of this method will look as follows: # noqa
        # prepare_inputs: bucket [D0, D1, D2] -> [D0, D1, D2, 0] (BS=4 bucket, 1 seq padding) # noqa
        # prepare_inputs: bucket [P0, P1, P2] -> [P0, P1], [P2] (BS=2 + BS=1 bucket, no seqs padding) # noqa
        # decode forward pass BS4 [D0, D1, D2, 0]
        # decode compute_logits BS4 [D0, D1, D2, 0]
        # decode sampler BS4 [D0, D1, D2, 0] -> [tokD0, tokD1, tokD2, 0]
        # prefill[iter 0] forward pass BS2 [P0, P1]
        # prefill[iter 0] compute_logits BS2 [P0, P1]
        # prefill[iter 0] sampler BS2 [P0, P1] -> [tokP0, tokP1]
        # prefill[iter 1] forward pass BS1 [P0, P1]
        # prefill[iter 1] compute_logits BS1 [P0, P1]
        # prefill[iter 1] sampler BS1 [P0, P1] -> [tokP2]
        # prefill concat sampler results [tokP0, tokP1], [tokP2] -> [tokP0, tokP1, tokP2] # noqa
        # Join the prefill and decode on device into [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] # noqa
        # Transfer [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] to CPU
        # On CPU, sanitize [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] -> [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2] # noqa
        # Return [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]

        # Example2: Same thing, but with max_prefill_batch_size=4:
        # prepare_inputs: bucket [D0, D1, D2] -> [D0, D1, D2, 0] (BS=4 bucket, 1 seq padding) # noqa
        # prepare_inputs: bucket [P0, P1, P2] -> [P0, P1, P2, 0] (BS=4 bucket, 1 seq padding) # noqa
        # decode forward pass BS4 [D0, D1, D2, 0]
        # decode compute_logits BS4 [D0, D1, D2, 0]
        # decode sampler BS4 [D0, D1, D2, 0] -> [tokD0, tokD1, tokD2, 0]
        # prefill[iter 0] forward pass BS4 [P0, P1, P2, 0]
        # prefill[iter 0] compute_logits BS4 [P0, P1, P2, 0]
        # prefill[iter 0] sampler BS4 [P0, P1, P2, 0] -> [tokP0, tokP1, tokP2, 0] # noqa
        # Join the prefill and decode on device into [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] # noqa
        # Transfer [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] to CPU
        # On CPU, sanitize [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] -> [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2] # noqa
        # Return [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]

        self.run_defragmenter(scheduler_output, warmup_mode)

        batch_changed = self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group() or warmup_mode:
                # Return empty ModelRunnerOuptut if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT
            # For D case, wait until kv finish load here
            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
        if self.input_batch.pooling_params:
            (input_ids, position_ids, num_scheduled_tokens, attn_metadata,
             total_scheduled_tokens) = self._prepare_inputs_for_pooling(scheduler_output)

            with set_forward_context(attn_metadata, self.vllm_config):
                hidden_states = self.model.forward(
                    input_ids=input_ids,
                    positions=position_ids,
                )

            flattened = hidden_states.view(-1, hidden_states.shape[-1])
            pooled_output = self._pool(
                flattened,
                total_scheduled_tokens,
                np.array(num_scheduled_tokens, dtype=np.int32),
            )
            return pooled_output
        # If necessary, swap decodes/prompts to have all decodes on the start

        ensure_decodes_first(self.input_batch)
        # Prepare prompts/decodes info
        pd_info = self._get_prompts_and_decodes(scheduler_output)
        num_decodes = len(pd_info.decode_req_ids)
        num_prefills = len(pd_info.prompt_req_ids)
        num_reqs = num_decodes + num_prefills
        with self.profiler.record_event('internal', 'prepare_input_tensors'):
            prefill_input_data, decode_input_data = self._prepare_inputs(scheduler_output, num_prefills, num_decodes,
                                                                         warmup_mode)
        prefill_data, \
            dummy_prefill_input_data_batches_across_dp = prefill_input_data
        num_pad_prefill_batch_across_dp = \
            0 if dummy_prefill_input_data_batches_across_dp is None \
            else len(dummy_prefill_input_data_batches_across_dp.request_ids)
        decode_data, dummy_decode_input_data_across_dp = decode_input_data
        #FIXME(kzawora): Currently there's no handling of logprobs. Fix that
        # later.
        prefill_sampled_token_ids = []
        prefill_sampled_requests = []
        decode_sampled_token_ids = []
        decode_sampled_requests = []
        #if not has_kv_transfer_group():
        #    assert not (num_prefills > 0 and num_decodes > 0)
        # skip kv_connector if dummy run
        if not warmup_mode:
            with set_forward_context(None, self.vllm_config):
                self.maybe_setup_kv_connector(scheduler_output)
        finished_sending, finished_recving = set(), set()

        # NOTE(Chendi): used by spec decode draft model, since we are doing
        # prefill one by one, so save hidden states as list
        non_flattened_hidden_states_prefills = []
        aux_hidden_states_prefills = []
        sample_hidden_states_prefills = []
        decode_sampled_token_ids_device = None
        # NOTE(tianmu-li): For structured output, combine logits before
        # postprocessing. Should it be done for all requests?
        structured_output = False
        spec_decode_num_tokens = None
        if scheduler_output.grammar_bitmask is not None:
            logits_prompt = []
            logits_decode = []
            structured_output = True

        if self.use_async_scheduling:
            invalid_req_indices = []
        ######################### PREFILLS #########################
        if num_prefills > 0:
            htorch.core.mark_step()
            for idx, (req_id, prompt_len, token_ids, position_ids, attn_metadata, logits_indices,
                      logits_requests) in enumerate(zip(*shallow_tuple(prefill_data))):

                inputs_embeds = None
                model_mm_kwargs = None
                if self.supports_mm_inputs:
                    # Run the multimodal encoder if any.
                    with self.profiler.record_event('internal', 'prepare_input_encoders'):
                        self._execute_mm_encoder(scheduler_output, req_id)

                    mm_embeds = self._gather_mm_embeddings(scheduler_output, req_id)
                    # TODO: Only get embeddings for valid token_ids. Ignore token_ids[<pad_idxs>] # noqa E501
                    # This may require moving multimodal input preps into _prepare_inputs,        # noqa E501
                    # to avoid padding issues.
                    inputs_embeds = self.model.get_input_embeddings(
                        input_ids=token_ids,
                        multimodal_embeddings=mm_embeds or None,
                    )

                    model_mm_kwargs = self._extract_mm_kwargs(scheduler_output)
                    model_mm_kwargs = MultiModalKwargs.as_kwargs(
                        model_mm_kwargs,
                        device=self.device,
                    )

                lora_mask, lora_logits_mask = self._configure_lora(token_ids, self.requests, req_id, True)

                self.event_start = self.profiler.get_timestamp_us()
                self.profiler.start("internal", "prefill")
                # NOTE(tianmu-li): Align behavior of incomplete prompt with gpu_model_runner
                # If logits_indices is smaller than req_id, the last request is a chunked prompt request that
                # hasn't finished in this step. We add the last token position to logits_indices to ensure
                # the last token of the chunk is sampled. This sampled token will be discarded later
                if logits_indices.shape[0] < len(req_id):
                    if structured_output or self.use_async_scheduling:
                        # When there are multiple requests in the batch (e.g. self.use_merged_prefill=True),
                        # the last token position is the sum of all prompt lengths - 1
                        # This logic also holds when there is only one request in the batch
                        logits_indices_append = torch.tensor([torch.sum(prompt_len) - 1],
                                                             device=token_ids.device,
                                                             dtype=torch.int32)
                        logits_indices = torch.cat([logits_indices, logits_indices_append])
                    if self.use_async_scheduling:
                        # Discard partial prefill logit for async scheduling
                        # Depends on 1 decode token/batch
                        prefill_start_idx = num_decodes
                        invalid_req_indices.append(prefill_start_idx + idx)
                htorch.core.mark_step()
                non_flattened_hidden_states, aux_hidden_states, \
                    sample_hidden_states, logits_device = \
                    self._execute_model_generic_omni(
                        token_ids, position_ids, attn_metadata, logits_indices,
                        self.kv_caches,
                        lora_logits_mask,
                        lora_mask,
                        inputs_embeds=inputs_embeds,
                        model_mm_kwargs=model_mm_kwargs,
                        warmup_mode=warmup_mode,)
                htorch.core.mark_step()
                non_flattened_hidden_states_prefills.append(non_flattened_hidden_states)
                if self.use_aux_hidden_state_outputs:
                    aux_hidden_states_prefills.append(aux_hidden_states)
                sample_hidden_states_prefills.append(sample_hidden_states)
                # Skip separate sampling for structured output
                if structured_output:
                    logits_prompt.append(logits_device)
                    prefill_sampled_requests.extend(logits_requests)
                else:
                    # If there are no logits, there is nothing to sample.
                    # This can happen with chunked prefill when a chunk does
                    # not complete the prompt and no logits are generated.
                    if logits_device.numel() > 0:
                        with self.profiler.record_event('internal', "sampler"):
                            sampler_output, sampling_metadata = self._run_sampling(batch_changed, logits_device, req_id,
                                                                                   logits_device.shape[0],
                                                                                   logits_requests)
                            prefill_sampled_token_ids.append(sampler_output.sampled_token_ids.flatten())
                            prefill_sampled_requests.extend(logits_requests)
                if self.is_driver_worker and self.profiler.enabled:
                    # Stop recording 'execute_model_generic' event
                    self.profiler.end()
                    event_end = self.profiler.get_timestamp_us()
                    counters = self.profiler_counter_helper.get_counter_dict(cache_config=self.cache_config,
                                                                             duration=event_end - self.event_start,
                                                                             seq_len=self._seq_len(attn_metadata),
                                                                             batch_size_padded=token_ids.size(0),
                                                                             real_batch_size=len(req_id),
                                                                             prompt_batch_idx=idx,
                                                                             is_prompt=True)
                    self.profiler.record_counter(self.event_start, counters)
            if not warmup_mode:
                self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = (self.get_finished_kv_transfers(scheduler_output))

            if self.is_driver_worker and self.profiler.enabled:
                self.profiler_counter_helper.reset_prompt_seq_stats()

        if num_pad_prefill_batch_across_dp > 0:
            for idx, (req_id, prompt_len, token_ids, position_ids, attn_metadata, logits_indices,
                      logits_requests) in enumerate(zip(*shallow_tuple(dummy_prefill_input_data_batches_across_dp))):
                htorch.core.mark_step()
                _, _, _, dummy_logits_device = \
                self._execute_model_generic_omni(
                    token_ids,
                    position_ids,
                    attn_metadata,
                    logits_indices,
                    self.kv_caches,
                    None,
                    None,
                    warmup_mode=warmup_mode)
                htorch.core.mark_step()

        ######################### DECODES #########################
        # Decodes run as one single batch with [padded_decode_bs, 1]
        if num_decodes > 0:
            assert decode_data is not None
            lora_mask, lora_logits_mask = self._configure_lora(decode_data.token_ids, self.requests,
                                                               pd_info.decode_req_ids, False)
            self.event_start = self.profiler.get_timestamp_us()
            self.profiler.start("internal", "decode")
            htorch.core.mark_step()
            non_flattened_hidden_states, aux_hidden_states, \
                sample_hidden_states, logits_device = \
                    self._execute_model_generic_omni(
                decode_data.token_ids,
                decode_data.position_ids,
                decode_data.attn_metadata,
                decode_data.logits_indices,
                self.kv_caches,
                lora_logits_mask,
                lora_mask,
                warmup_mode=warmup_mode)
            htorch.core.mark_step()

            if structured_output:
                logits_decode.append(logits_device[:num_decodes])
                decode_sampled_requests.extend(self.input_batch.req_ids[:num_decodes])
            else:
                with self.profiler.record_event('internal', "sampler"):
                    ##### Sampling Start #####
                    spec_decode_metadata = decode_data.spec_decode_metadata
                    sampler_output, sampling_metadata = self._run_sampling(
                        batch_changed, logits_device
                        if spec_decode_metadata is None else logits_device[spec_decode_metadata.bonus_logits_indices],
                        pd_info.decode_req_ids, logits_device.shape[0])

                    if spec_decode_metadata is None:
                        decode_sampled_token_ids.append(sampler_output.sampled_token_ids.flatten())
                    else:
                        # Handling spec decode sampling.
                        bonus_token_ids = \
                            sampler_output.sampled_token_ids.squeeze()
                        target_logits = logits_device[spec_decode_metadata.target_logits_indices]

                        output_token_ids = self.rejection_sampler(
                            spec_decode_metadata,
                            None,  # draft_probs
                            target_logits,
                            bonus_token_ids,
                            sampling_metadata,
                        )
                        decode_sampled_token_ids = \
                            self.rejection_sampler.parse_output(
                                output_token_ids,
                                self.input_batch.vocab_size,
                        )
                        # convert decode_sampled_token_ids as list of tensor
                        spec_decode_num_tokens = [len(v) for v in decode_sampled_token_ids]
                        decode_sampled_token_ids = [
                            torch.tensor(v, device="cpu").int() for v in decode_sampled_token_ids
                        ]
                        decode_sampled_token_ids_device = \
                            output_token_ids.to("hpu", non_blocking=True)
                    decode_sampled_requests.extend(self.input_batch.req_ids[:num_decodes])
                    ##### Sampling End #####

            if self.is_driver_worker and self.profiler.enabled:
                # Stop recording 'execute_model' event
                self.profiler.end()
                event_end = self.profiler.get_timestamp_us()
                counters = self.profiler_counter_helper.get_counter_dict(
                    cache_config=self.cache_config,
                    duration=event_end - self.event_start,
                    seq_len=self._seq_len(decode_data.attn_metadata),
                    batch_size_padded= \
                        decode_data.token_ids.size(0), # type: ignore
                    real_batch_size=decode_data.num_decodes,
                    prompt_batch_idx=None,
                    is_prompt=False)
                self.profiler.record_counter(self.event_start, counters)

        elif dummy_decode_input_data_across_dp is not None:
            htorch.core.mark_step()
            _, _, _, dummy_logits_device = self._execute_model_generic_omni(dummy_decode_input_data_across_dp.token_ids,
                                                                       dummy_decode_input_data_across_dp.position_ids,
                                                                       dummy_decode_input_data_across_dp.attn_metadata,
                                                                       dummy_decode_input_data_across_dp.logits_indices,
                                                                       self.kv_caches,
                                                                       None,
                                                                       None,
                                                                       warmup_mode=warmup_mode)
            htorch.core.mark_step()

        if structured_output:
            # Scheduler places cached before prompt
            logits_combined = logits_decode + logits_prompt
            logits = torch.cat(logits_combined, dim=0)
            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                self.apply_grammar_bitmask(scheduler_output, logits)
            sampler_output, _sampling_metadata = self._run_sampling(batch_changed, logits,
                                                                    pd_info.prompt_req_ids + pd_info.decode_req_ids,
                                                                    logits.shape[0])
            # Deal with the case of incomplete prompt
            for i in range(logits.shape[0] - num_decodes):
                prefill_sampled_token_ids.append(sampler_output.sampled_token_ids[num_decodes + i].flatten())
            decode_sampled_token_ids.append(sampler_output.sampled_token_ids[:num_decodes].flatten())
        elif self.use_async_scheduling:
            # For async scheduling: keep tokens on HPU and avoid CPU sync
            # Concatenate decode and prefill tokens on HPU
            if decode_sampled_token_ids or prefill_sampled_token_ids:
                decode_sampled_token_ids = [tensor[:num_decodes] for tensor in decode_sampled_token_ids]
                # Note: this will cause an issue with the current spec decode impl, as they are on different devices
                sampled_token_ids = torch.cat(decode_sampled_token_ids + prefill_sampled_token_ids).view(-1, 1)
            else:
                sampled_token_ids = torch.empty((0, 1), dtype=torch.int32, device=self.device)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = \
            self.input_batch.req_id_to_index.copy()

        max_req_index = max(self.input_batch.req_id_to_index.values())
        postprocessed_sampled_token_ids: list[list[int]] = [[] for _ in range(max_req_index + 1)]
        if self.use_async_scheduling:
            self.input_batch.prev_sampled_token_ids = sampled_token_ids.flatten()
            # self.input_batch.prev_sampled_token_ids_invalid_indices
            invalid_req_indices_set = set(invalid_req_indices)
            self.input_batch.prev_sampled_token_ids_invalid_indices = \
                invalid_req_indices_set
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids) if i not in invalid_req_indices_set
            }
            # For the output, postprocessed_sampled_token_ids will be filled during serialization
        else:
            prefill_sampled_token_ids_device = prefill_sampled_token_ids
            # From this point onward, all operations are done on CPU.
            # We already have tokens. Let's copy the data to
            # CPU as is, and then discard padded tokens.
            with self.profiler.record_event('internal', "sampler_postprocessing"):
                prefill_sampled_token_ids = [tensor.cpu() for tensor in prefill_sampled_token_ids]
                if spec_decode_num_tokens is not None:
                    decode_sampled_token_ids = [tensor.cpu() for tensor in decode_sampled_token_ids]
                else:
                    decode_sampled_token_ids = [tensor.cpu()[:num_decodes] for tensor in decode_sampled_token_ids]
                sampled_token_ids_list = []
                # When there is no prompt or decode, skip concat to avoid error
                if (len(prefill_sampled_token_ids) + len(decode_sampled_token_ids)) > 0:
                    sampled_token_ids_list = torch.cat(decode_sampled_token_ids + prefill_sampled_token_ids).tolist()
                sampled_token_requests = \
                    decode_sampled_requests + prefill_sampled_requests
                max_req_index = max(self.input_batch.req_id_to_index.values())
                # NOTE(Chendi): in post-processing, spec_decode might
                # return more than 1 token during decode.
                start_idx = 0
                for i, req_id in enumerate(sampled_token_requests):
                    num_tokens = spec_decode_num_tokens[
                        i] if spec_decode_num_tokens is not None and i < num_decodes else 1
                    postprocessed_sampled_token_ids[
                        self.input_batch.req_id_to_index[req_id]] += sampled_token_ids_list[start_idx:start_idx +
                                                                                            num_tokens]
                    start_idx += num_tokens

        ################## RETURN ##################
        # NOTE(kzawora): idk what happens if part of batch doesn't have logprobs

        ######### UPDATE REQUEST STATE WITH GENERATED TOKENS #########
        for req_id in self.input_batch.req_ids[:num_reqs]:
            req_state = self.requests[req_id]
            i = self.input_batch.req_id_to_index[req_id]
            seq_len = (req_state.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id])
            token_ids = postprocessed_sampled_token_ids[i]
            num_tokens = len(token_ids)
            self.input_batch.token_ids_cpu[i, seq_len:seq_len + num_tokens] = token_ids
            self.input_batch.num_tokens[i] += len(token_ids)

        # NOTE(chendi): enable cache based on PR(#20291)
        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        for req_idx, sampled_ids in enumerate(postprocessed_sampled_token_ids[:num_reqs]):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            # NOTE(adobrzyn): assert for full max prompt length including
            # max_model_len and one token that's going to be generated
            # especially needed for biggest prompt in warm-up phase
            full_max_prompt = self.max_model_len + 1
            assert end_idx <= full_max_prompt, ("Sampled token IDs exceed the max model length. "
                                                f"Total number of tokens: {end_idx} > max_model_len: "
                                                f"{full_max_prompt}")

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        ################## Spec Decode ##################
        # Now, we will call drafter to propose draft token ids
        if self.speculative_config:
            self._draft_token_ids = self.propose_draft_token_ids(
                scheduler_output, postprocessed_sampled_token_ids, prefill_sampled_token_ids_device,
                decode_sampled_token_ids_device, sampling_metadata, non_flattened_hidden_states, sample_hidden_states,
                aux_hidden_states, non_flattened_hidden_states_prefills, sample_hidden_states_prefills,
                aux_hidden_states_prefills, num_decodes, prefill_data if num_prefills > 0 else None,
                decode_data if num_decodes > 0 else None)
        ################## Spec Decode end ##################

        # Create output.
        all_req_ids = pd_info.decode_req_ids + pd_info.prompt_req_ids
        # prompt_logprobs_dict: dict[
        #    str, Optional[LogprobsTensors]] = self._get_prompt_logprobs_dict(
        #        prefill_hidden_states_device, scheduler_output)
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        all_req_ids = pd_info.decode_req_ids + pd_info.prompt_req_ids
        logprobs = None

        if self.use_async_scheduling:
            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,  # CHECK
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=postprocessed_sampled_token_ids,
                logprobs=logprobs,
                prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
                pooler_output=[],
            )
            return AsyncHPUModelRunnerOutput(
                model_runner_output=model_runner_output,
                sampled_token_ids=sampled_token_ids,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
            )

        model_runner_output = OmniModelRunnerOutput(
            req_ids=all_req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=postprocessed_sampled_token_ids,
            logprobs=logprobs,
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
            pooler_output=[],
            kv_connector_output=KVConnectorOutput(
                finished_sending=finished_sending,
                finished_recving=finished_recving,
            ),
            num_nans_in_logits=None,
            )
        if has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()

        return model_runner_output



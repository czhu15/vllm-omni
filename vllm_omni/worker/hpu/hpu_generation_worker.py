import gc
import os

import torch
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import GiB_bytes, MemorySnapshot
from vllm.v1.utils import report_usage_stats
from vllm_gaudi.v1.worker.hpu_worker import HPUWorker
from vllm_gaudi.v1.worker.hpu_worker import init_worker_distributed_environment

from vllm_omni.worker.hpu.hpu_generation_model_runner import HPUGenerationModelRunner

from vllm.logger import init_logger

logger = init_logger(__name__)

class HPUGenerationWorker(HPUWorker):
    """HPU Worker for Generation model (non-autoregressive waveform generation).

    Usage in stage config:
        worker_cls: "vllm_omni.worker.hpu_generation_model_runner.HPUGenerationModelRunner"
    """

    def init_device(self):
        """Initialize CUDA device and distributed environment."""
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.cuda.empty_cache()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
            if self.init_snapshot.free_memory < self.requested_memory:

                def GiB(b):
                    return round(b / GiB_bytes, 2)

                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        elif self.device_config.device.type == "hpu":
            # TODO(czhu15) move this duplicated code to HPUWorker
            logger.info(f"set_device, {self.local_rank=}")
            self.device = torch.device("hpu")
            current_platform.set_device(self.local_rank)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after HCCL is initialized
            gc.collect()

            # take current memory snapshot
            # TODO(czhu15) implement hpu version of memory snapshot
            # self.init_snapshot = MemorySnapshot()
            # self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
            # if self.init_snapshot.free_memory < self.requested_memory:

            #     def GiB(b):
            #         return round(b / GiB_bytes, 2)

            #     raise ValueError(
            #         f"Free memory on device "
            #         f"({GiB(self.init_snapshot.free_memory)}/"
            #         f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
            #         f"is less than desired GPU memory utilization "
            #         f"({self.cache_config.gpu_memory_utilization}, "
            #         f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
            #         f"utilization or reduce GPU memory used by other processes."
            #     )
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Construct the model runner
        self.model_runner: HPUGenerationModelRunner = HPUGenerationModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

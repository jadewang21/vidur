from math import ceil
from typing import List, Optional

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.logger import init_logger

logger = init_logger(__name__)

class VLLMReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_requests: List[Request] = []
        self._num_running_batches = 0
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

    def schedule(self) -> Optional[Batch]:
        # 直接调用 _get_next_batch，确保预占请求被调度
        batch = self._get_next_batch()
        if batch:
            self._batch_counter += 1
            self._num_running_batches += 1
        return batch

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        completed_requests = [request for request in batch.requests if request.completed]
        logger.info(f"Batch {batch.id} completed requests: {len(completed_requests)}")

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)
        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return

        self.allocate(request.id, 1)

    def _get_next_batch(self) -> Optional[Batch]:
        requests = []
        num_tokens = []
        num_batch_tokens = 0

        # 优先调度预占请求
        if self._preempted_requests:
            logger.info(f"Rescheduling {len(self._preempted_requests)} preempted requests")
            self._preempted_requests.sort(key=lambda r: r.arrived_at)
            while self._preempted_requests:
                if len(requests) >= self._max_micro_batch_size:
                    break
                request = self._preempted_requests[0]
                if not (self._config.num_blocks - self._num_allocated_blocks >= 1):
                    break
                self._preempted_requests.pop(0)
                self._allocate_request(request)
                next_num_tokens = self._get_request_next_num_tokens(request)
                requests.append(request)
                num_tokens.append(next_num_tokens)
                num_batch_tokens += next_num_tokens

        # 调度新请求
        while self._request_queue:
            request = self._request_queue[0]
            next_num_tokens = self._get_request_next_num_tokens(request)

            if not self._can_allocate_request(request):
                break

            new_num_tokens = num_tokens + [next_num_tokens]
            new_num_batch_tokens = len(new_num_tokens) * max(new_num_tokens)
            if new_num_batch_tokens > self._config.max_tokens_in_batch:
                break

            # 放宽 allocation_map 限制，允许更多请求
            if len(self._allocation_map) >= self._config.batch_size_cap * 2:  # 放宽到 2 倍
                break

            if len(requests) >= self._max_micro_batch_size:
                break

            request = self._request_queue.pop(0)
            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            num_batch_tokens += next_num_tokens

        if requests:
            return Batch(self._replica_id, requests, num_tokens)

        return None
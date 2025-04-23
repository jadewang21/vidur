from math import ceil
from typing import List

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)

from vidur.scheduler.replica_stage_scheduler.replica_stage_schduler import ReplicaStageScheduler  # 导入 ReplicaStageScheduler

class VLLMReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        self._simulator = kwargs.pop('simulator', None) # 使用 pop 从 kwargs 中取出 simulator，这样它就不会被传给 super
        #super().__init__(*args, **kwargs)
        super().__init__(*args, simulator=self._simulator, **kwargs)
        
        #self._simulator = kwargs.get('simulator')   # 获取 simulator 参数
        #self._execution_time_predictor = kwargs.get('execution_time_predictor')
        #self._simulator = kwargs.pop('simulator', None)  # 移除 simulator 参数
        self._preempted_requests: List[Request] = []
        self._num_running_batches = 0
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )
        # # 创建 ReplicaStageScheduler 实例
        # self._stage_schedulers = [
        #     ReplicaStageScheduler(
        #         replica_id=self._replica_id,
        #         stage_id=i,
        #         is_last_stage=(i == self._num_stages - 1),
        #         execution_time_predictor=self._execution_time_predictor,
        #         #simulator=self._simulator,  # 传递 simulator
        #     )
        #     for i in range(self._num_stages)
        # ]

    def on_request_arrival(self, current_time: float, request: Request) -> None:
        print(f"VLLMReplicaScheduler: Scheduling request {request.id} at time {current_time}, arrived_at: {request.arrived_at}")
        request.scheduled_at = current_time
        request.scheduling_delay = current_time - request.arrived_at
        request.scheduled = True
        print(f"VLLMReplicaScheduler: Request {request.id} scheduled_at: {request.scheduled_at}, scheduling_delay: {request.scheduling_delay}")
        self._request_queue.append(request)
        print(f"VLLMReplicaScheduler: Request queue length: {len(self._request_queue)}")

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            can_allocate = (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )
            print(f"Checking allocation for request {request.id}: num_required_blocks={num_required_blocks}, can_allocate={can_allocate}")
            return can_allocate

        # vllm requires at least one block to be available
        can_allocate = self._config.num_blocks - self._num_allocated_blocks >= 1
        print(f"Checking allocation for request {request.id} (existing): can_allocate={can_allocate}")
        return can_allocate

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            print(f"Allocating {num_required_blocks} blocks for request {request.id}")
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)
        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return

        print(f"Allocating 1 additional block for request {request.id}")
        self.allocate(request.id, 1)

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        num_batch_tokens = 0

        print(f"Getting next batch, request queue length: {len(self._request_queue)}")
        while self._request_queue:
            request = self._request_queue[0]

            next_num_tokens = self._get_request_next_num_tokens(request)
            print(f"Request {request.id}: next_num_tokens={next_num_tokens}")

            if not self._can_allocate_request(request):
                print(f"Cannot allocate request {request.id}, breaking")
                break

            new_num_tokens = num_tokens + [next_num_tokens]
            new_num_batch_tokens = len(new_num_tokens) * max(new_num_tokens)
            print(f"New num batch tokens: {new_num_batch_tokens}, max_tokens_in_batch: {self._config.max_tokens_in_batch}")
            if new_num_batch_tokens > self._config.max_tokens_in_batch:
                print(f"Batch tokens exceed max_tokens_in_batch, breaking")
                break

            print(f"Allocation map length: {len(self._allocation_map)}, batch_size_cap: {self._config.batch_size_cap}")
            if len(self._allocation_map) == self._config.batch_size_cap:
                print(f"Allocation map full, breaking")
                break

            print(f"Current batch size: {len(requests)}, max_micro_batch_size: {self._max_micro_batch_size}")
            if len(requests) == self._max_micro_batch_size:
                print(f"Batch size reached max_micro_batch_size, breaking")
                break

            request = self._request_queue.pop(0)

            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            num_batch_tokens += next_num_tokens

        if requests:
            print(f"Returning batch with {len(requests)} requests")
            return Batch(self._replica_id, requests, num_tokens)

        # Safer to sort preempted_requests to maintain FIFO order
        self._preempted_requests.sort(key=lambda r: r.arrived_at)
        print(f"Checking preempted requests, length: {len(self._preempted_requests)}")
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                print(f"Batch size reached max_micro_batch_size for preempted requests, breaking")
                break

            request = self._preempted_requests.pop(0)

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue = [victim_request] + self._request_queue
                    print(f"Preempted request {victim_request.id} restarted and added to queue")
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue = [request] + self._request_queue
                    print(f"Request {request.id} restarted and added to queue")
                    break
            else:
                self._allocate_request(request)
                next_num_tokens = self._get_request_next_num_tokens(request)
                requests.append(request)
                num_tokens.append(next_num_tokens)

        if not requests:
            print("No batch created")
            return None

        print(f"Returning batch with {len(requests)} requests from preempted requests")
        return Batch(self._replica_id, requests, num_tokens)
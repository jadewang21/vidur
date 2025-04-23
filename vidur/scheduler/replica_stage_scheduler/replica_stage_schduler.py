from typing import Tuple

from vidur.entities import Batch, BatchStage, ExecutionTime
from vidur.execution_time_predictor import BaseExecutionTimePredictor

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReplicaStageScheduler:
    def __init__(
        self,
        replica_id: int,
        stage_id: int,
        is_last_stage: bool,
        execution_time_predictor: BaseExecutionTimePredictor,
        simulator=None,  # 添加 simulator 参数
    ) -> None:
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage
        self._execution_time_predictor = execution_time_predictor
        self._simulator = simulator  # 存储 simulator 实例
        self._batch_queue = []
        self._is_busy = False

    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    def is_empty(self) -> bool:
        return len(self._batch_queue) == 0

    def add_batch(self, batch: Batch) -> None:
        print(f"Adding batch to queue, batch requests: {[r.id for r in batch.requests]}")
        self._batch_queue.append(batch)
        print(f"Batch queue length: {len(self._batch_queue)}")

    def on_stage_end(self) -> None:
        self._is_busy = False

    def on_schedule(self) -> Tuple[Batch, BatchStage, ExecutionTime]:
        if self._is_busy or not self._batch_queue:
            print(f"Cannot schedule: is_busy={self._is_busy}, batch_queue_length={len(self._batch_queue)}")
            return None, None, None

        self._is_busy = True
        batch = self._batch_queue.pop(0)
        #print(f"Executing batch at stage {self._stage_id}, batch requests: {[r.id for r in batch.requests]}")
        request_ids = [r.id for r in batch.requests]
        is_prefill = [not r.is_prefill_complete for r in batch.requests] # 示例，具体看 batch/request 属性
        logger.debug(f"Simulator time: {self._simulator._time}, Stage {self._stage_id}: Predicting execution time for batch {batch.id} (requests: {request_ids}, num_tokens: {batch.num_tokens}, is_prefill: {is_prefill})")
        #logger.debug(f"Simulator time: {self._simulator.current_time}, Stage {self._stage_id}: Predicting execution time for batch {batch.id} (requests: {request_ids}, num_tokens: {batch.num_tokens}, is_prefill: {is_prefill})")
        execution_time = self._execution_time_predictor.get_execution_time(
            batch,
            self._stage_id,
        )
        total_execution_time = execution_time.total_time
        model_execution_time = execution_time.model_time
        
        #logger.debug(f"Simulator time: {self._simulator.current_time}, Stage {self._stage_id}: Predicted execution time for batch {batch.id}: total={total_execution_time}, model={model_execution_time}")
        logger.debug(f"Simulator time: {self._simulator._time}, Stage {self._stage_id}: Predicted execution time for batch {batch.id}: total={total_execution_time}, model={model_execution_time}")
        
        batch_stage = BatchStage(
            batch.id,
            self._replica_id,
            self._stage_id,
            total_execution_time,
            model_execution_time,
            batch.requests,
            batch.num_tokens,
        )

        # for request in batch.requests:
        #     request.latest_stage_scheduled_at = self._simulator.current_time
        #     print(f"Request {request.id} latest_stage_scheduled_at: {request.latest_stage_scheduled_at}, num_processed_tokens: {request.num_processed_tokens}")
        #     if request.num_processed_tokens >= request.num_prefill_tokens + request.num_decode_tokens:
        #         request.completed_at = self._simulator.current_time
        #         request.completed = True
        #         print(f"Request {request.id} completed at time {request.completed_at}")
        logger.info(f"Simulator time: {self._simulator._time}, Stage {self._stage_id}: Predicted execution time for batch {batch.id} (requests: {request_ids}): Total={total_execution_time:.6f}, Model={model_execution_time:.6f}")
        return batch, batch_stage, execution_time
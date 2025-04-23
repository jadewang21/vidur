import pandas as pd
from typing import List
from vidur.request_generator.base_request_generator import BaseRequestGenerator
from vidur.types import RequestIntervalGeneratorType
from vidur.utils.random import set_seeds
from vidur.entities import Request
from vidur.request_generator.request_length_generator_registry import (
    RequestLengthGeneratorRegistry,
)
from vidur.request_generator.request_interval_generator_registry import (
    RequestIntervalGeneratorRegistry,
)

class SyntheticRequestGenerator(BaseRequestGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.request_length_generator = RequestLengthGeneratorRegistry.get(
            self.config.length_generator_config.get_type(),
            self.config.length_generator_config
        )
        self.request_interval_generator = RequestIntervalGeneratorRegistry.get(
            self.config.interval_generator_config.get_type(),
            self.config.interval_generator_config
        )

        # 打印 request_metrics_adjusted.csv 的前几行，确认文件内容
        #df = pd.read_csv(self.config.interval_generator_config.trace_file)
        #print("First 5 rows of request_metrics_adjusted.csv:")
        #print(df.head())

    def _generate_next_request(self, last_arrived_at: float) -> Request:
        inter_request_time = (
            self.request_interval_generator.get_next_inter_request_time()
        )
        print(f"Inter-request time: {inter_request_time}")
        if inter_request_time is None:
            return None
        arrived_at = last_arrived_at + inter_request_time
        print(f"last_arrived_at: {last_arrived_at}, inter_request_time: {inter_request_time}, arrived_at: {arrived_at}")

        (
            prefill_tokens,
            decode_tokens,
        ) = self.request_length_generator.get_next_num_tokens()
        print(f"prefill_tokens: {prefill_tokens}, decode_tokens: {decode_tokens}")

        if prefill_tokens is None or decode_tokens is None:
            return None

        return Request(
            arrived_at=arrived_at,
            num_prefill_tokens=int(prefill_tokens),
            num_decode_tokens=int(decode_tokens),
        )

    def _generate_requests(self) -> List[Request]:
        requests = []

        current_time = 0

        # first priority is duration
        if self.config.duration is not None:
            while current_time < self.config.duration:
                request = self._generate_next_request(current_time)
                print(f"Generated request at time {current_time}: {request}")
                if request is None:
                    print("Request is None, breaking loop")
                    break
                current_time = request.arrived_at
                requests.append(request)
        elif self.config.num_requests is not None:
            for i in range(self.config.num_requests):
                request = self._generate_next_request(current_time)
                print(f"Request {i+1}/{self.config.num_requests}: {request}")
                if request is None:
                    print(f"Request {i+1} is None, breaking loop")
                    break
                current_time = request.arrived_at
                requests.append(request)
        else:
            assert (
                self.config.interval_generator_config.get_type()
                == RequestIntervalGeneratorType.TRACE
            )

            while True:
                request = self._generate_next_request(current_time)
                print(f"Generated request at time {current_time}: {request}")
                if request is None:
                    print("Request is None, breaking loop")
                    break
                current_time = request.arrived_at
                requests.append(request)

        return requests

    def generate_requests(self) -> List[Request]:
        assert (
            self.config.duration
            or self.config.num_requests
            or self.config.interval_generator_config.get_type()
            == RequestIntervalGeneratorType.TRACE
        )

        set_seeds(self.config.seed)

        requests = self._generate_requests()

        # 移除重新排序逻辑，确保按文件顺序
        # requests.sort(key=lambda x: x.arrived_at)
        # remove any requests that arrived after the time limit
        if self.config.duration is not None:
            requests = [
                request
                for request in requests
                if request.arrived_at < self.config.duration
            ]

        return requests
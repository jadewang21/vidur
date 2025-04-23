import logging
import pandas as pd
from vidur.config import TraceRequestIntervalGeneratorConfig
from vidur.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)

logger = logging.getLogger(__name__)

class TraceRequestIntervalGenerator(BaseRequestIntervalGenerator):
    def __init__(self, config: TraceRequestIntervalGeneratorConfig):
        super().__init__(config)

        # load into a pd dataframe
        self.trace_df = pd.read_csv(config.trace_file)

        self.trace_df["arrival_time"] = pd.to_datetime(self.trace_df["arrival_time"])
        print("Before filtering - First 5 rows of trace_df:")
        print(self.trace_df.head())

        # restrict trace_df to be a subset of rows that have the same date
        self.trace_df = self.trace_df[
            (self.trace_df["arrival_time"] > config.start_time)
            & (self.trace_df["arrival_time"] < config.end_time)
        ]

        print("After filtering - First 5 rows of trace_df:")
        print(self.trace_df.head())

        # change back to seconds (keep floating point precision)
        self.trace_df["arrival_time"] = (
            self.trace_df["arrival_time"] - self.trace_df["arrival_time"].min()
        ).dt.total_seconds()

        print("After converting to seconds - First 5 rows of trace_df:")
        print(self.trace_df.head())

        # rescale the time to change QPS
        self.trace_df["arrival_time"] = (
            self.trace_df["arrival_time"] * config.time_scale_factor
        )

        print("After rescaling - First 5 rows of trace_df:")
        print(self.trace_df.head())

        # compute the inter-request time
        self.trace_df["inter_request_time"] = self.trace_df["arrival_time"].diff()

        print("After computing inter_request_time - First 5 rows of trace_df:")
        print(self.trace_df.head())

        self.next_request_idx = 0  # 从 0 开始

        logger.info(
            f"Loaded interval trace file {config.trace_file} with {len(self.trace_df)} requests"
        )

    def get_next_inter_request_time(self) -> float:
        if self.next_request_idx >= len(self.trace_df):
            return None

        if self.next_request_idx == 0:
            # 第一个请求的 inter_request_time 为 0
            self.next_request_idx += 1
            print("next_request_idx: 0, inter_request_time: 0.0 (first request)")
            return 0.0

        inter_request_time = self.trace_df.iloc[self.next_request_idx][
            "inter_request_time"
        ]
        print(f"next_request_idx: {self.next_request_idx}, inter_request_time: {inter_request_time}")
        self.next_request_idx += 1

        return inter_request_time
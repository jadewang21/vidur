import logging
from typing import Tuple
import numpy as np
import pandas as pd
from vidur.config import TraceRequestLengthGeneratorConfig
from vidur.request_generator.base_request_length_generator import BaseRequestLengthGenerator

logger = logging.getLogger(__name__)

class TraceRequestLengthGenerator(BaseRequestLengthGenerator):
    def __init__(self, config: TraceRequestLengthGeneratorConfig):
        super().__init__(config)

        self.trace_df = pd.read_csv(config.trace_file)
        print("TraceRequestLengthGenerator - Before processing - First 5 rows of trace_df:")
        print(self.trace_df.head())

        # Ensure arrival_time is in datetime format and sort by arrival_time
        self.trace_df["arrival_time"] = pd.to_datetime(self.trace_df["arrival_time"])
        self.trace_df.sort_values(by="arrival_time", inplace=True)
        print("TraceRequestLengthGenerator - After sorting by arrival_time - First 5 rows of trace_df:")
        print(self.trace_df.head())

        # scale prefill and decode tokens
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] * config.prefill_scale_factor
        )
        self.trace_df["num_decode_tokens"] = (
            self.trace_df["num_decode_tokens"] * config.decode_scale_factor
        )

        # make sure all the prefill and decode counts are integers
        self.trace_df["num_prefill_tokens"] = self.trace_df[
            "num_prefill_tokens"
        ].astype(int)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].astype(
            int
        )

        # make sure the total does not exceed the max tokens, adjust the prefill tokens if needed
        total_tokens = (
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
        )
        diff_tokens = total_tokens - config.max_tokens
        diff_tokens = diff_tokens.clip(lower=0)

        # deduct the diff tokens from the prefill and decode tokens proportionally
        prefill_tokens_ratio = self.trace_df["num_prefill_tokens"] / total_tokens
        decode_tokens_ratio = self.trace_df["num_decode_tokens"] / total_tokens

        self.trace_df["num_prefill_tokens"] -= (
            np.ceil(diff_tokens * prefill_tokens_ratio)
        ).astype(int)

        self.trace_df["num_decode_tokens"] -= (
            np.ceil(diff_tokens * decode_tokens_ratio)
        ).astype(int)

        # make sure that there is at least one prefill and decode token
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].clip(
            lower=1
        )
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].clip(
            lower=1
        )

        assert all(
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
            <= self.config.max_tokens
        )

        assert all(self.trace_df["num_prefill_tokens"] > 0)
        assert all(self.trace_df["num_decode_tokens"] > 0)

        print("TraceRequestLengthGenerator - After processing - First 5 rows of trace_df:")
        print(self.trace_df.head())

        # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
        pd_ratio = (
            self.trace_df["num_prefill_tokens"] / self.trace_df["num_decode_tokens"]
        )
        logger.info(
            f"Loaded request length trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        pd_distribution = pd_ratio.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        logger.debug(f"Prompt/decode token ratio stats\n: {pd_distribution}")

        # Removed random shuffling to maintain order
        self.next_request_idx = 0

    def get_next_num_tokens(self) -> Tuple[float, float]:
        if self.next_request_idx >= len(self.trace_df):
            return None, None

        row = self.trace_df.iloc[self.next_request_idx]
        prefill_tokens = row["num_prefill_tokens"]
        decode_tokens = row["num_decode_tokens"]
        print(f"TraceRequestLengthGenerator - next_request_idx: {self.next_request_idx}, prefill_tokens: {prefill_tokens}, decode_tokens: {decode_tokens}")
        self.next_request_idx += 1

        return prefill_tokens, decode_tokens
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Decodes the raw float32 logits produced by cadence_executor_runner_sim
(via --dump-output) for the SmolLM-135M .pte, and prints the predicted
next token after the real (non-padding) part of the prompt.
"""

import argparse
import logging

import numpy as np
from transformers import AutoTokenizer

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="examples/cadence/models/smollm135",
        help="Directory with tokenizer.json/tokenizer_config.json.",
    )
    parser.add_argument(
        "--logits_bin",
        default="smollm135-out-0.bin",
        help="Raw float32 output tensor written by --dump-output.",
    )
    parser.add_argument(
        "--seq_len", type=int, default=32, help="Must match the exported model shape."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=49152, help="Must match the model config."
    )
    parser.add_argument(
        "--prompt_len",
        type=int,
        required=True,
        help="Number of real (non-padding) tokens, printed by smollm_135_prepare_input.py.",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of top candidates to print."
    )
    args = parser.parse_args()

    logits = np.fromfile(args.logits_bin, dtype=np.float32).reshape(
        1, args.seq_len, args.vocab_size
    )

    # Logits at the last real token predict the next token in the sequence.
    next_token_logits = logits[0, args.prompt_len - 1]
    top_ids = np.argsort(next_token_logits)[::-1][: args.top_k]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    logging.info("Top %d next-token predictions:", args.top_k)
    for token_id in top_ids:
        token_str = tokenizer.decode([token_id])
        logging.info(
            "  id=%d logit=%.3f token=%r", token_id, next_token_logits[token_id], token_str
        )

    predicted_id = int(top_ids[0])
    logging.info("Predicted next token: %r", tokenizer.decode([predicted_id]))


if __name__ == "__main__":
    main()

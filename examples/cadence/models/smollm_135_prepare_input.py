# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenizes a real text prompt into the fixed-shape (1, seq_len) int64
input_ids tensor expected by the SmolLM-135M .pte exported via
smollm_135.py, and writes it as a raw .bin file consumable by
cadence_executor_runner_sim's --inputs flag.
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
        "--prompt", default="The capital of France is", help="Text prompt to encode."
    )
    parser.add_argument(
        "--seq_len", type=int, default=32, help="Must match the exported model shape."
    )
    parser.add_argument(
        "--out",
        default="input_ids-0.bin",
        help="Output raw int64 .bin file for --inputs.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    token_ids = tokenizer(args.prompt)["input_ids"]
    prompt_len = len(token_ids)
    assert (
        prompt_len <= args.seq_len
    ), f"Prompt has {prompt_len} tokens; --seq_len={args.seq_len} is too small."

    # Right-pad with eos/pad so the tensor exactly matches the exported shape.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    padded = token_ids + [pad_id] * (args.seq_len - prompt_len)

    np.array(padded, dtype=np.int64).tofile(args.out)

    logging.info("Prompt: %r", args.prompt)
    logging.info("Token ids (%d real, padded to %d): %s", prompt_len, args.seq_len, padded)
    logging.info("Wrote %s (last real token at index %d).", args.out, prompt_len - 1)
    logging.info(
        "Run with: --inputs=%s and note --prompt_len=%d for decoding.",
        args.out,
        prompt_len,
    )


if __name__ == "__main__":
    main()

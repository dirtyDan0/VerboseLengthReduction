# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from pathlib import Path
from verl import DataProto


class Dumper:

    def __init__(self, tokenizer, file_path=None, exp_name=None):
        self.tokenizer = tokenizer
        self.file_path = Path(file_path)
        self.exp_name = exp_name
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def dump(self, data: DataProto, step: int, split="train"):
        prompt_ids = data.batch['prompts']
        response_ids = data.batch['responses']
        attention_mask = data.batch['attention_mask']

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        scores = data.batch['token_level_rewards'].sum(dim=-1)
        results = data.non_tensor_batch["result"]
        verbose_lens = data.non_tensor_batch["verbose_len"]
        indexs = data.non_tensor_batch["index"]
        truncateds = data.non_tensor_batch["truncated"]

        with self.file_path.open("a", encoding="utf-8") as f:
            for i in range(len(data)):
                row = {
                    "exp_name": self.exp_name,
                    "split": split,
                    "step": int(step),
                    "index": int(indexs[i]),
                    "result": results[i],
                    "length": int(valid_response_lengths[i]),
                    "score": float(scores[i]),
                    "response": responses_str[i],
                    "verbose_len": int(verbose_lens[i]),
                    "truncated": bool(truncateds[i])
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

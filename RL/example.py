# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
python examples/scripts/ppo.py \
    --log_with=wandb
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline


from trlmain.trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
from trlmain.trl.core import LengthSampler, set_seed
from trlmain.trl.import_utils import is_npu_available, is_xpu_available
from trlmain.trl.trainer.ppo_config import PPOConfig
from trlmain.trl.trainer.ppo_trainer import PPOTrainer

import json
from AlignScoresrc.alignscore import AlignScore
scorer = AlignScore(model='./AlignScore-base/Roberta-base/', batch_size=8, device='cuda:0', ckpt_path='AlignScore-base/AlignScore-base.ckpt', evaluation_mode='nli_sp')

import yake
language = "en"
max_ngram_size = 2
deduplication_threshold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 10

custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)

def repeat_tensors(tensor_list, repeat):
    repeated_list = []
    for tensor in tensor_list:
        repeated_list.extend([tensor] * repeat)
    return repeated_list


def extract_digits(s):
    start_index = 0
    for i, char in enumerate(s):
        if char.isdigit():
            start_index = i
            break

    end_index = 0
    for i, char in enumerate(s[::-1]):
        if char.isdigit():
            end_index = len(s) - i
            break

    return s[start_index:end_index]


def extract_digits_list(s_list):
    new_list = []
    for i_1 in range(len(s_list)):
        new_list.append(extract_digits(s_list[i_1]))
    return new_list


knowledge_base_file = open("knowledge_base/knowledge_base_doc.json")
knowledge_base_content = json.loads(knowledge_base_file.read())


def docid_2_doc(doc_ids):
    text = []
#     print(doc_ids)
#     print('**************')
    for i in range(len(doc_ids)):
        correct = 1
        if len(doc_ids[i]) == 0:
            correct = 0
        for num in doc_ids[i]:
            if num > '9' or num < '0':
                correct = 0
                
        if correct == 0:
            idx = 0
        else:
#             print(doc_ids[i])
            idx = int(doc_ids[i])
            if idx > 2000 or idx < 0:
                idx = 0
        doc = knowledge_base_content[idx]['text']
        text.append(doc)

    return text


tqdm.pandas()

import os
os.environ['CURL_CA_BUNDLE'] = ''

@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser((ScriptArguments, PPOConfig))
args, ppo_config = parser.parse_args_into_dataclasses()

## some setting by hsw
ppo_config.model_name = './../DSI-QG-main/checkpoint'

args.use_seq2seq = True


# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead

dataset = load_dataset("json", data_files='dataset/che_training_dataset.json', split='train')

tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample


dataset = dataset.map(tokenize, batched=False)
dataset.set_format(type="torch")


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(ppo_config.model_name, trust_remote_code=args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

model = trl_model_class.from_pretrained(
    ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)


tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

################################################################
# docid generation constrain, we only generate integer docids. Added by hsw
SPIECE_UNDERLINE = "▁"
INT_TOKEN_IDS = []
for token, id in tokenizer.get_vocab().items():
    if token[0] == SPIECE_UNDERLINE:
        if token[1:].isdigit():
            INT_TOKEN_IDS.append(id)
    if token == SPIECE_UNDERLINE:
        INT_TOKEN_IDS.append(id)
    elif token.isdigit():
        INT_TOKEN_IDS.append(id)
INT_TOKEN_IDS.append(tokenizer.eos_token_id)


def sample_rewards(arr, n):
    return arr[::n]

def restrict_decode_vocab(batch_idx, prefix_beam):
    return INT_TOKEN_IDS


################################################################

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
# device = ppo_trainer.accelerator.device
# if ppo_trainer.accelerator.num_processes == 1:
#     if is_xpu_available():
#         device = "xpu:0"
#     elif is_npu_available():
#         device = "npu:0"
#     else:
#         device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
# ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
# task, model_name = ppo_config.reward_model.split(":")
# if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
#     with ds_plugin.zero3_init_context_manager(enable=False):
#         sentiment_pipe = pipeline(task, model=model_name, device=device)
# else:
#     sentiment_pipe = pipeline(task, model=model_name, device=device)

# # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
# if sentiment_pipe.tokenizer.pad_token_id is None:
#     sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

# if sentiment_pipe.model.config.pad_token_id is None:
#     sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

for epochs in range(1):

    for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    #     if(_epoch < 2):
    #         continue
        # Remember to change the model.generate the same
        Num_of_Retrieved_Document = 16
#         ppo_config.batch_size *= Num_of_Retrieved_Document
#         ppo_config.mini_batch_size *= Num_of_Retrieved_Document

        query_tensors = batch["input_ids"]
        query_tensors_for_step = repeat_tensors(query_tensors, Num_of_Retrieved_Document)
        print('')
        # print('-------------------')
        # print(query_tensors)
        # print(query_tensors_for_step)

        # change the query_tensors'form to the form suitable for generative retrieval
        max_length = max(len(tensor) for tensor in query_tensors)
        for i, tensor in enumerate(query_tensors):
            padding_length = max_length - len(tensor) + 1
            padding_values = [1] + [0] * (padding_length - 1)
            query_tensors[i] = torch.cat((tensor, torch.tensor(padding_values, device=tensor.device)))

        query_tensors = torch.stack(query_tensors)
        # print(query_tensors)

        # Get response from retriever
        response_tensors, resp_scores, ref_response_tensors, ref_scores = ppo_trainer.generate(
            restrict_decode_vocab,
            query_tensors,
            return_prompt=False,
            generate_ref_response=True,
            return_scores=True,
            **generation_kwargs
        )

        # to be changed to decode
        batch["response"] = tokenizer.batch_decode(response_tensors)

        # true_response = tokenizer.decode(response_tensors[1])
        # print(true_response)

        response_tensors = [tensor for tensor in response_tensors]

        response_tensors_for_step = []
        for tensor in response_tensors:
            non_zero_indices = (tensor != 0).nonzero()
            if len(non_zero_indices) > 0:
                last_non_zero_idx = non_zero_indices[-1].item()
                response_tensors_for_step.append(tensor[:last_non_zero_idx])

        response_for_retrieval = extract_digits_list(batch['response'])

    #     print(batch['query'])
    #     print(batch['label'])
    #     # print(batch['response'])
    #     print(response_for_retrieval)
    #     # print(response_tensors_for_step)
    #     print('----------------------------')
    #     print('-----------------------------------------------------------------------')

        ### rewards computation
        # Here we want the feedback from AlignScore as reward:
        # As part of the reward, we prefer the Retrieved document that can prove of disprove the claim, so the score close to 1 and 0 is prefered.
        # To avoid those completely irrelevant document, which also will result in a zero,
        # we detect how many keywords in  claim is covered by document, if not many, the document will get bad rewards
        claims = []
        contexts = docid_2_doc(response_for_retrieval)

        for i in range(len(batch['query'])):
            claim = batch['query'][i]
            for j in range(Num_of_Retrieved_Document):
                claims.append(claim)

        score = scorer.score(contexts=contexts, claims=claims)

        rewards = []

        for i in range(len(score)):
            reward_hallu = score[i]

            keywords = custom_kw_extractor.extract_keywords(claims[i])
            keywords = [t[0] for t in keywords]

            num_entail = 0
            for j in range(len(keywords)):
                keyword = keywords[j]
                if keyword in contexts[i]:
                    num_entail += 1
                    if j < 5:
                        num_entail+=1


            reward_overlap = num_entail / 15

            reward = reward_hallu - 0.2 * (1 - reward_overlap)
            reward = torch.tensor(reward)

            rewards.append(reward)


        # a local bias according to the query itself
        def find_median(arr):
            n = len(arr) // Num_of_Retrieved_Document
            median_array = []
            for i in range(n):
                group = sorted(arr[i*Num_of_Retrieved_Document : (i+1)*Num_of_Retrieved_Document])
                median = group[int(Num_of_Retrieved_Document/2)] if len(group) % 2 != 0 else (group[int(Num_of_Retrieved_Document/2)-1] + group[int(Num_of_Retrieved_Document/2)]) / 2
                median_array.append(median)
            return median_array

        def subtract_median(arr, median_array):
            result = []
            for i in range(len(arr)):

                idx = i // Num_of_Retrieved_Document
                result.append(arr[i] - median_array[idx])
            return result

        median_reward = find_median(rewards)
    #     print(median_reward)

        rewards = subtract_median(rewards, median_reward)

#         print(rewards)

        del query_tensors
        del response_tensors
        del ref_response_tensors

        query_tensors_for_step = take_every_nth_element(query_tensors_for_step, Num_of_Retrieved_Document)
        response_tensors_for_step = take_every_nth_element(response_tensors_for_step, Num_of_Retrieved_Document)
        rewards = sample_rewards(rewards, Num_of_Retrieved_Document)

        # Run PPO step
        stats = ppo_trainer.step(query_tensors_for_step, response_tensors_for_step, rewards)
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
#         ppo_config.mini_batch_size /= Num_of_Retrieved_Document
#         ppo_config.mini_batch_size = int(ppo_config.mini_batch_size)
#         ppo_config.batch_size /= Num_of_Retrieved_Document
#         ppo_config.batch_size = int(ppo_config.batch_size)


    torch.save(model.state_dict(), "./../DSI/RL_checkpoint/pytorch_model.bin")
    tokenizer.save_pretrained('./../DSI/RL_checkpoint')
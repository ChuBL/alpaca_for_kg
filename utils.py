import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

from openai import AzureOpenAI
import tqdm
import copy

StrOrOpenAIObject = Union[str, dict]

azure_endpoint = "https://ai-openaiwebui300260177230.services.ai.azure.com"
api_version = "2025-01-01-preview"
deployment_name = "gpt-4o-mini"

client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=os.getenv("OPENAI_GPT4o_API_KEY"),
    api_version=api_version,
)


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logprobs: Optional[int] = None


def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="gpt-4o-mini",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
):
    is_single_prompt = isinstance(prompts, (str, dict, list)) and not isinstance(prompts[0], (list, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)

        while True:
            try:
                shared_kwargs = dict(
                    model=model_name,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )

                if isinstance(prompt_batch[0], list):
                    messages = prompt_batch[0]  # 已是消息列表时直接使用
                elif isinstance(prompt_batch[0], str):
                    messages = [{"role": "user", "content": prompt} for prompt in prompt_batch]
                else:
                    raise ValueError(f"Unexpected prompt format: {type(prompt_batch[0])}")

                completion_batch = client.chat.completions.create(
                    messages=messages,
                    model=model_name,  # 必须明确指定
                )
                choices = completion_batch.choices

                for choice in choices:
                    choice.total_tokens = completion_batch.usage.total_tokens
                    choice.text = choice.message.content
                completions.extend(choices)
                break
            except Exception as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        (completions,) = completions
    return completions


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

import numpy as np
from typing import Any, Dict, List, Mapping, Optional, Callable, Tuple
from torch.utils.data import Dataset
from torchtune.modules.tokenizers import Tokenizer
from torchtune.data import Message, ChatFormat, CROSS_ENTROPY_IGNORE_IDX, Message, validate_messages
from torchtune.datasets import ChatDataset
from datasets import load_dataset

class CustomChatFormat(ChatFormat):
    system = ""
    user = ""
    assistant = ""

    @classmethod
    def format(
        cls,
        sample: List[Message],
    ) -> List[Message]:
        return sample

class CustomChatDataset(Dataset):
    """
    Class that supports any custom dataset with multiturn conversations.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> foreach turn{format into template -> tokenize}

    If the column/key names differ from the expected names in the `ChatFormat`,
    then the `column_map` argument can be used to provide this mapping.

    Use `convert_to_messages` to prepare your dataset into the llama conversation format
    and roles:
        [
            {
                "role": <system|user|assistant>,
                "content": <message>,
            },
            ...
        ]

    This class supports multi-turn conversations. If a tokenizer sample with multiple
    turns does not fit within `max_seq_len` then it is truncated.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        convert_to_messages (Callable[[Mapping[str, Any]], List[Message]]): function that keys into the desired field in the sample
            and converts to a list of `Messages` that follows the llama format with the expected keys
        chat_format (ChatFormat): template used to format the chat. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use `column_map` to map them.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        source: str,
        convert_to_messages: Callable[[Mapping[str, Any]], List[Message]],
        chat_format: Optional[ChatFormat] = None,
        max_seq_len: int,
        train_on_input: bool = False,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._convert_to_messages = convert_to_messages
        self.chat_format = chat_format
        self.max_seq_len = max_seq_len
        self.train_on_input = train_on_input

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        messages = self._convert_to_messages(sample, self.train_on_input)
        if self.chat_format is not None:
            messages = self.chat_format.format(messages)
        validate_messages(messages)
        tokens, mask = self._tokenizer.tokenize_messages(
            messages, max_seq_len=self.max_seq_len
        )
        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels)

        return tokens, labels
    
chat_format = CustomChatFormat()

def _converter(sample: Mapping[str, Any], train_on_input: bool=False) -> List[Message]:
    system_msg = sample.get('instruction', '')
    text_msgs = [sample['user'], sample['assistant']]
    
    is_user = True
    # Multi-turn dataset
    if system_msg:
        msgs = [
            Message(
                role="system",
                content=system_msg,
                masked=True, # Mask if not training on prompt
            )
        ]
    else:
        msgs = []
        
    for m in text_msgs:
        msgs.append(
            Message(
                role="user" if is_user else "assistant",
                content=m,
                masked=is_user,
            )
        )
        is_user = not is_user

    return msgs

def custom_chat_dataset(
    tokenizer: Tokenizer,
    source: str,
    max_seq_len: Optional[int] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> CustomChatDataset:

    return CustomChatDataset(
        tokenizer=tokenizer,
        source=source,
        convert_to_messages=_converter,
        # Llama3 does not need a chat format
        chat_format=chat_format,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )
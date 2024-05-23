# https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_text_completion.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict, Any, List, Mapping, Tuple

from torchtune.data import AlpacaInstructTemplate, InstructTemplate, truncate
from torchtune.datasets import InstructDataset
from torchtune.models.llama3 import llama3_tokenizer
from torchtune.modules.tokenizers import Tokenizer
from datasets import load_dataset
from torch.utils.data import Dataset



def koalpaca_dataset(
    tokenizer: Tokenizer=llama3_tokenizer,
    source: str="beomi/KoAlpaca-v1.1a",
    template: InstructTemplate=AlpacaInstructTemplate,
    train_on_input: bool=True,
    max_seq_len: Optional[int]=2048,
    split: str="train",
):
    '''
    Build a KoAlpaca Dataset.

    Args:
        tokenizer (Tokenizer, optional): A Tokenizer which is used to encode or decode. Defaults to llama3_tokenizer.
        source (str, optional): An url or a path of data source. Defaults to "beomi/KoAlpaca-v1.1a".
        template (InstructTemplate, optional): A Template of instruction and output. Defaults to AlpacaInstructTemplate.
        train_on_input (bool, optional): ... Defaults to True.
        max_seq_len (Optional[int], optional): The length of max sequence. Defaults to 2048.
        split (str, optional): ... Defaults to "train".

    Returns:
        _type_: _description_
    '''
    return InstructDataset(
        tokenizer=tokenizer,
        source=source,
        template=template,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        split=split,
    )

class TextCompletionDataset(Dataset):
    """
    Freeform dataset for any unstructured text corpus. Quickly load any dataset
    from Hugging Face or local disk and tokenize it for your model.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an ``encode`` and ``decode`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        column (Optional[str]): name of column in the sample that contains the text data. This is typically required
            for Hugging Face datasets or tabular data, but can be omitted for local datasets. Default is None.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        column: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.max_seq_len = max_seq_len
        self._column = column

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        prompt = sample[self._column] if self._column is not None else sample
        tokens = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=True)

        # Truncate if needed, but don't coerce EOS id
        if self.max_seq_len is not None:
            tokens = truncate(tokens, self.max_seq_len - 1)

        # No need to offset labels by 1 - happens in the recipe
        labels = tokens.copy()

        return {"tokens": tokens, "labels": labels}


def text_completion_dataset(
    tokenizer: Tokenizer,
    source: str,
    column: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> TextCompletionDataset:
    """
    Build a configurable freeform text dataset with instruction prompts. This method should be
    used to configure a custom text dataset from the yaml config instead of
    using `TextDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an ``encode`` and ``decode`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        column (Optional[str]): name of column in the sample that contains the text data. This is typically required
            for Hugging Face datasets or tabular data, but can be omitted for local datasets. Default is None.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Examples:
        >>> from torchtune.datasets import text_completion_dataset
        >>> dataset = text_completion_dataset(
        ...   tokenizer=tokenizer,
        ...   source="allenai/c4",
        ...   column="text",
        ...   max_seq_len=2096,
        ...   data_dir="realnewslike",
        ... )

    This can also be accomplished via the yaml config::

        dataset:
            _component_: torchtune.datasets.text_completion_dataset
            source: allenai/c4
            column: text
            max_seq_len: 2096
            data_dir: realnewslike

    Returns:
        TextCompletionDataset: the configured :class:`~torchtune.datasets.TextCompletionDataset`
    """
    return TextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        column=column,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )
    
def kowiki_dataset(
    tokenizer=llama3_tokenizer,
    source='csv',
    column='text',
    max_seq_len=2048,
    **load_dataset_kwargs,
):
    return TextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        column=column,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )
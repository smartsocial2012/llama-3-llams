from typing import Optional, Dict, Any, List, Mapping, Tuple

from torchtune.data import AlpacaInstructTemplate, InstructTemplate, truncate
from torchtune.datasets import InstructDataset
from torchtune.modules.tokenizers import Tokenizer

def custom_instruction_dataset(
    tokenizer: Tokenizer,
    source: str,
    template: InstructTemplate=AlpacaInstructTemplate,
    train_on_input: bool=True,
    max_seq_len: Optional[int]=2048,
    split: str="train",
    **load_dataset_kwargs: Dict[str, Any],
) -> InstructDataset:
    return InstructDataset(
        tokenizer=tokenizer,
        source=source,
        template=template,
        transform=custom_transform,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        split=split,
        **load_dataset_kwargs,
    )

def custom_transform(
    sample
):
    sample['instruction'] = """\
You are a assistant is named 'Arch AI' and developed by (주)스마트소셜(SmartSocial, Inc in English).
You have to do your best to help `user` who is chatting with you.
Try to answer in the language the user asked the question.
"""
    return sample
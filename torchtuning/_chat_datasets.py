from typing import Any, Dict, List, Mapping, Optional
from torchtune.modules.tokenizers import Tokenizer
from torchtune.data import Message, ChatFormat
from torchtune.datasets import ChatDataset

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

chat_format = CustomChatFormat()

def _converter(sample: Mapping[str, Any], train_on_input: bool=False) -> List[Message]:
    system_msg = sample.get('system', '')
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
) -> ChatDataset:

    return ChatDataset(
        tokenizer=tokenizer,
        source=source,
        convert_to_messages=_converter,
        # Llama3 does not need a chat format
        chat_format=chat_format,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )
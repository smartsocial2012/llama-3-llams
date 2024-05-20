from torchtune.data import AlpacaInstructTemplate
from torchtune.datasets import InstructDataset
from torchtune.models.llama3 import llama3_tokenizer

def koalpaca_dataset(
    tokenizer=llama3_tokenizer,
    source="beomi/KoAlpaca-v1.1a",
    template=AlpacaInstructTemplate,
    train_on_input=True,
    max_seq_len=2048,
    split="train",
):

    return InstructDataset(
        tokenizer=tokenizer,
        source=source,
        template=template,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        split=split,
    )
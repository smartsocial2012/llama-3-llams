# Environments
```
conda install conda-forge::transformers
pip install -U "huggingface_hub[cli]"
pip install -U datasets # fsspec의 업데이트로 인한 충돌이 있었다고 함. 최신 release인 dataset==2.14.6으로 업데이트 함으로써 문제 해결 가능
pip install accelerate evaluate peft trl scikit-learn
pip install torchtune
```

# Command Line
## Fine-tuning
```
tune run --nproc_per_node 3 lora_finetune_distributed --config configs/llams_base.yaml
tune run --nproc_per_node 3 lora_finetune_distributed --config configs/llams_instruct.yaml
tune run --nproc_per_node 3 lora_finetune_distributed --config configs/llams_instr_train_1k_conversations.yaml
```
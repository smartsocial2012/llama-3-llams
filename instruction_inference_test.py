import torch, torchtune, yaml
from torchtune import config, utils
from torchtune.data import AlpacaInstructTemplate
from generate import InferenceRecipe
from omegaconf import DictConfig
from matplotlib import pyplot as plt
import pandas as pd, os
import itertools

INSTRUCTION = '''You are a assistant is named 'Arch AI' and developed by (주)스마트소셜(SmartSocial, Inc in English).
You have to do your best to help `user` who is chatting with you.
Try to answer in the language the user asked the question.'''

checkpoints_1k = 'checkpoints/Llama-3-LLaMS-inst-from-base/instrunction_template'
checkpoints_2k = 'checkpoints/Llama-3-LLaMS-inst-from-base/instrunction_template_2k_instsep'

def inputting(input_):
    transform = AlpacaInstructTemplate.format
    ret = transform(
        {
            'instruction': INSTRUCTION,
            'input': input_
        }
    )
    return ret

config_base = './configs/llams_inst_generation.yaml'
config_lists = {
    # 'origin': {
    #     'checkpoint_dir': 'checkpoints/Meta-Llama-3-8B-Instruct/original',
    #     'checkpoint_files': ['consolidated.00.pth'],
    #     'output_dir': 'checkpoints/Meta-Llama-3-8B-Instruct/original',
    # },
    'from_base': {
        'checkpoint_dir': 'checkpoints/Llama-3-LLaMS-inst-from-base-start',
        'checkpoint_files': ['meta_model_0.pt'],
        'output_dir': 'checkpoints/Llama-3-LLaMS-inst-from-base-start'
    },
}

for x in [checkpoints_1k, checkpoints_2k]:
    for xx in range(20):
        checkpoint_file = f'meta_model_{xx}.pt'
        config_lists['/'.join([x, checkpoint_file])] = {
            'checkpoint_dir': x,
            'checkpoint_files': [checkpoint_file],
            'output_dir': x
        }
        
results = {}
for config_key, config_value in config_lists.items():
    with open(config_base) as f:
        config_dict = DictConfig(yaml.safe_load(f))
    for k, v in config_value.items():
        config_dict.checkpointer[k] = v
    config_dict.prompt = inputting('Tell me about Plato.')
    
    results_for_topktemp = {}
    for topk, temperature in itertools.product([300, 1], [0, .6, .8]):
        config_dict.topk = topk
        config_dict.temperature =temperature
        
        a = InferenceRecipe(config_dict)
        a.setup(config_dict)

        res = a.generate(config_dict)
        results_for_topktemp[f'topk: {topk}, temp: {temperature}'] = res[len(config_dict.prompt):]

        del a
        torch.cuda.empty_cache()
    results[config_key] = results_for_topktemp

df = pd.DataFrame.from_dict(results)
df.T.to_csv('inference_test_inst_token_2048_eng_columnwise.csv')

if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference test for the instruction model.')
    parser.add_argument('--instruction', type=str, default=INSTRUCTION)
    parser.add_argument('--input', type=str, default='친구야, 안녕?')
    
    args = parser.parse_args()


import os
from pathlib import Path

import torch
from PIL import Image
from lavis.common.registry import registry
from lavis.models import load_model_and_preprocess
from lavis.datasets.builders import load_dataset

from flan_t5_gpt3_caption import FlanGPTCaption

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    registry.mapping['paths']['cache_root'] = Path.cwd() / 'export'

    dataset = load_dataset(name='ok_vqa', cfg_path=None)['test']

    sample = dataset.annotation[0]

    model, img_processor_dict, text_processor_dict = load_model_and_preprocess(
        name='blip2_t5_gpt3_caption', 
        model_type='pretrain_flant5xl',
        is_eval=True,
        device=device
    )
    img_processor = img_processor_dict['eval']
    text_processor = text_processor_dict['eval']
    
    # TODO: Dit moet handiger kunnen maar hun gare voorbeeld waarbij de dataset
    # images instantieert in de dataset ipv paden naar images klopt niet?
    img_path = f'{dataset.vis_root}{sample["image"]}'
    sample['image'] = img_processor(
        Image.open(f'{img_path}').convert('RGB')
    ).unsqueeze(0).to(device)
    sample['text_input'] = text_processor(
        sample['question']
    )

    result = model.predict_answers(
        samples=sample,
        inference_method='generate',
        num_beams=5,
        max_len=10,
        min_len=1,
        prompt="Question: {} Short answer:"
    )

    print(
f'''
Question: {sample['text_input']}
Answer: {result}
Image: {img_path}
''')
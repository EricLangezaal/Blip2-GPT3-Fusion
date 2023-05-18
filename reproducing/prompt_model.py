import os
from pathlib import Path
from collections import defaultdict

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
    annotations = dataset.annotation[0:4]

    model, img_processor_dict, text_processor_dict = load_model_and_preprocess(
        name='blip2_t5_gpt3_caption', 
        model_type='pretrain_flant5xl',
        is_eval=True,
        device=device
    )
    img_processor = img_processor_dict['eval']
    text_processor = text_processor_dict['eval']
    
    samples = {
        'image': None,
        'text_input': [],
        'prompt': ''
    }

    for annotation in annotations:
        img_path = f'{dataset.vis_root}{annotation["image"]}'
        image = img_processor(
            Image.open(f'{img_path}').convert('RGB')
        ).unsqueeze(0).to(device)
        text_input = text_processor(
            annotation['question']
        )

        if samples['image'] is None:
            samples['image'] = image
        else:
            samples['image'] = torch.cat((samples['image'], image))
        
        samples['text_input'].append(text_input)

    result = model.predict_answers(
        samples=samples,
        inference_method='generate',
        num_beams=5,
        max_len=10,
        min_len=1,
        prompt="Question: {} Short answer:"
    )

    print(
f'''
Question: {samples['text_input']}
Answer: {result}
''')
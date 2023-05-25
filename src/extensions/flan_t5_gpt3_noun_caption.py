import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
import time

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration

import openai
from openai.error import RateLimitError, APIError, ServiceUnavailableError

from extensions.gpt_utils import *
from reproducing.flan_t5_int8 import Blip2T5int8

@registry.register_model("blip2_t5_gpt3_noun_caption")
class FlanGPTNounCaption(Blip2T5int8):


    def __init__(
        self,
        openai_api_key="",
        verbose=False,
        **args,
    ):
        super().__init__(**args)
        
        openai.api_key = openai_api_key
        self.verbose = verbose

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        try:
            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]

            samples["prompt"] = prompt_question(samples["text_input"])
            output_text = self.generate(
                samples=samples,
                max_length=max_len,
                min_length=min_len, 
                length_penalty=length_penalty, 
                top_p = 1.0
            )

            if self._apply_lemmatizer:
                output_text = self._lemmatize(output_text)          

            # GENERATION OF OBJECT DESCRIPTION
            paper_prompt = "a photo of"

            # input change to batched because noun different for each question
            paper_prompt = [paper_prompt] * len(samples["text_input"])

            # let GPT pick the most important noun in questions
            nouns = []
            for question in samples["text_input"]:
                picked_noun = noun_gpt(question, temperature=0)
                picked_noun = get_single_answer(picked_noun)
                print(f"GPT picked '{picked_noun}' from '{question}'")
                nouns.append(picked_noun)
            
            # create BLIP-2 generated context for GPT3 
            noun_prompts = [f"the {noun} can be described as" for noun in nouns]
            prompt_batches = []
            prompt_batches.append(paper_prompt)
            prompt_batches.append(noun_prompts)

            contexts = []
            for prompt_batch in prompt_batches: 
                samples['prompt'] = prompt_batch

                answer_to_gpt_embed = self.generate(
                        samples=samples,
                        use_nucleus_sampling=False,
                        num_beams=5,
                        max_length=20,
                        min_length=1,
                        #top_p=0.9,
                        repetition_penalty=2.0,
                        length_penalty=1.5,
                        num_captions=1, 
                        temperature=0
                        )
                context = [f"{prompt} {answer}" for prompt, answer in zip(prompt_batch, answer_to_gpt_embed)]
                contexts.append(context)
    
            contexts = list(zip(*contexts))

            # let GPT3 answer from context
            gpt_answers_batch = []
            for context, org_question, original_answer in zip(contexts, samples["text_input"], output_text):
                print('BLIP generated context for GPT: ', '. '.join(context))
                gpt_answers_batch.append(context_gpt(context, org_question, original_answer, temperature=0, verbose=self.verbose))

            if self._apply_lemmatizer:
                gpt_answers_batch = self._lemmatize(gpt_answers_batch)

            if self.verbose:
                print('---------------------New batch---------------------')
                for context, org_question, gpt_answer, original_answer in zip(contexts, samples['text_input'], gpt_answers_batch, output_text):
                    print('Original question: ', org_question)
                    print('Blip answer', original_answer)
                    print('GPT answer: ', gpt_answer)
                    print('\n ')
                    print('context for gpt', context)
                    print('-----------------------------------')
            
            return gpt_answers_batch
        
        except (RateLimitError, APIError, ServiceUnavailableError):
            time.sleep(5)
            return self.predict_answers(
                samples,
                num_beams,
                inference_method,
                max_len,
                min_len,
                num_ans_candidates,
                answer_list,
                prompt,
                length_penalty,
                **kwargs
            )


    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        openai_api_key = cfg.get("openai_api_key", "")
        verbose = cfg.get("verbose", False)

        model = cls(
            openai_api_key=openai_api_key,
            verbose=verbose,
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer
        )

        model.load_checkpoint_from_config(cfg)

        return model
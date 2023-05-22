"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration

import openai
from openai.error import RateLimitError, APIError

from gpt_utils import *
from flan_t5_gpt3_int8 import Blip2T5int8
import time

@registry.register_model("blip2_t5_gpt3_caption")
class FlanGPTCaption(Blip2T5int8):

    def __init__(
        self,
        openai_api_key="",
        verbose=False,
        **args
    ):
        super().__init__(**args)

        openai.api_key = openai_api_key
        self.verbose = verbose

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        prompt = samples["prompt"]
        if isinstance(prompt, str):
            prompt = [prompt] * image.size(0)
        
        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        outputs = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        output_text = self.t5_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return output_text

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
                gpt_answers_batch.append(context_gpt(context, org_question, original_answer, temperature=0))

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
        
        except (RateLimitError, APIError):
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
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
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
import time

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration

import openai
from openai.error import RateLimitError, APIError, ServiceUnavailableError

from extensions.gpt_utils import *
from reproducing.flan_t5_int8 import Blip2T5int8

@registry.register_model("blip2_t5_gpt3_summarize")
class FlanGPTSummarize(Blip2T5int8):


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

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            prompted_text_input = prompt_question(text_input)

            input_tokens = self.t5_tokenizer(
                prompted_text_input, padding="longest", return_tensors="pt"
            ).to(image.device)
            

            # GENERATION OF REGULAR ANSWER
            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            if self._apply_lemmatizer:
                output_text = self._lemmatize(output_text)
            
            #################### ADDED PART  ########################################
            # GENERATION OF OBJECT DESCRIPTION
            gpt_questions = gpt_generate_questions(text_input)

            listed_answers = []
            for batch in list(zip(*gpt_questions)):
                description_tokens = self.t5_tokenizer(
                    prompt_question(list(batch)), padding="longest", return_tensors="pt").to(image.device)
                encoder_atts_new = torch.cat([atts_t5, description_tokens.attention_mask], dim=1)

                description_embeds = self.t5_model.encoder.embed_tokens(description_tokens.input_ids)
                description_embeds = torch.cat([inputs_t5, description_embeds], dim=1)
                

                answer_to_gpt_embed = self.t5_model.generate(
                    inputs_embeds=description_embeds,
                    attention_mask=encoder_atts_new,
                    do_sample=False,
                    num_beams=num_beams,
                    # TODO: OPTIMISE SETTINGS FOR PHOTO DESCRIPTION!
                    max_new_tokens=15,
                    min_length=1,
                    repetition_penalty = 1.5,
                    # -1 (default) gives 1 word answers, '2' gives sentences
                    length_penalty=1,
                )
                answer_to_gpt_question = self.t5_tokenizer.batch_decode(
                    answer_to_gpt_embed, skip_special_tokens=True
                )

                listed_answers.append(answer_to_gpt_question)
            
            listed_answers = list(zip(*listed_answers))

            gpt_summarised_batch = []
            for questions, answers, org_question, org_answer in zip(gpt_questions, listed_answers, text_input, output_text):
                gpt_summarised_batch.append(summarized_gpt(questions, answers, org_question, org_answer))

            if self._apply_lemmatizer:
                gpt_summarised_batch = self._lemmatize(gpt_summarised_batch)
            
            if self.verbose:
                print('new batch')
                for gpt_summarized, questions, answers, org_question, org_answer in zip(gpt_summarised_batch, gpt_questions, listed_answers, text_input, output_text):
                    print("Original question: ", org_question)
                    print("Original answer: ", org_answer)
                    print("GPT generated questions:", questions)
                    print("THe answers to those GPT questions:", answers)
                    print("FINAL answer: ", gpt_summarized, "\n")        
            ################ ADDED PART ######################
            return gpt_summarised_batch
        
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
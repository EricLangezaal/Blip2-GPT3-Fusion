import torch
import openai


def BLIP2_image_captioning(image, model, processor, device, prompt=None):
    """"
    Takes as input an image, model, processor, device and (optionally) a prompt and
    returns a (prompted) caption for this image, created by BLIP2.
    """

    if prompt:
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
    else:
        inputs = processor(image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return generated_text


def prompt_llm(prompt, max_tokens=64, temperature=0, stop=None):
  """
  Helper function for prompting the GPT3 language model
  """

  response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
  return response["choices"][0]["text"].strip()


def GPT3_image_captioning(image, model, processor,device, prompt=None):
    """"
    Takes as input an image, model, processor, device and (optionally) a prompt and
    returns a (prompted) caption for this image, improved by GPT3.
    """
    generated_text = BLIP2_image_captioning(image, model, processor, device, prompt)

    prompt_GPT = f"Instruction: augment or improve the answer. If the given answer is factually wrong, correct it \
    in a similar answer style. \n Context: {prompt}. {generated_text}"

    return prompt_llm(prompt_GPT)



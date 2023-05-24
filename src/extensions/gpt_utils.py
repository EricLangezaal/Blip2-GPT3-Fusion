import openai
from pathlib import Path
import re
import os

def prompt_chat_gpt(prompt, max_tokens=64, temperature=0.7, stop=None):
  """
  Helper function for prompting the GPT3 chat-based language model
  """


  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content":  "You are a helpful assistant that generates open-ended questions that extract visual information"},
        {"role": "user", "content": f"Someone asked you '{prompt}'. List the three best distinct open-ended questions you would like to be answered \
         by a visual information retrieval system, such that you can best answer '{prompt}'. Only list open-ended questions that can be answered \
            without modifying the image. I want only the distinct questions, don't say anything else."}
        ],
    max_tokens = 150,
    temperature=temperature)

  return response["choices"][0]['message']["content"].strip()

def gpt_generate_questions(input_questions, temperature=0.7):

    gpt_questions = []
    for question in input_questions:
       questions = prompt_chat_gpt(question, temperature=temperature)
       questions = questions.split('\n')

       parsed_questions = []
       for q in questions:
          match = re.search("[a-zA-Z].*", q)
          if match:
             parsed_questions.append(match.group())
          else:
             parsed_questions.append(q)

       gpt_questions.append(questions)
       
    return gpt_questions

def prompt_question(questions):
    if isinstance(questions, list):
      return [f"Question: {q} Short answer:" for q in questions]
    return f"Question: {questions} Short answer:"

def summarized_gpt(questions, answers, original_question, original_answer, temperature=0.2):
    """
    Helper function for prompting the GPT3 chat-based language model
    """
    messages = [
        {"role": "system", "content":  "You are truthful assistant that answers the main question in one or two words. You can utilise the information from multiple partial questions that have been answered."},
        {"role": "user", "content": f"The main question is '{original_question}'. The original answer from the visual question answering model was '{original_answer}'"}
        ]
    
    for question, answer in zip(questions, answers):
       messages.append({"role": "user", "content": f"We asked: {question}"})
       messages.append({"role": "user", "content": f"The answer: {answer}"})

    messages.extend([{"role": "user", "content": f"Please answer the main question in one or two words: '{original_question}'. You can either repeat the original answer '{original_answer}', or improve it if necessary, still using only one or two words."}])

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages,
      max_tokens = 10,
      temperature=temperature)

    match = re.search("[a-zA-Z].*[a-zA-Z]", response["choices"][0]['message']["content"].strip())
    if match:
      return match.group()
    else:
      return response["choices"][0]['message']["content"].strip()
    
    
def get_gpt_unknowns():
  dirname = Path(os.path.dirname(os.path.realpath(__file__)))
  path = dirname / "configs/gpt_unknown_answers.txt"
  with path.open(mode="r") as file:
     gpt_unknowns = [w.lower() for w in file.read().splitlines()]
  return gpt_unknowns

def context_gpt(all_info, original_question, original_answer, temperature=0, verbose=False):
    """
    Helper function for prompting the GPT3 chat-based language model
    """
    messages = [
        {"role": "system", "content":  "Give a single answer to the question in one or two words by using the context"},
        ]
    ex_q = "What is the name of the famous dreamworks animated film where this animal was voiced by chris rock?"
    ex_q2 = "What is the name of the bridge in the background?"
    ex_q3 = "What are the most popular countries for this sport?"

    messages.append({"role": "user", "content": f"Context: A photo of a zebra. Question: {ex_q}."})
    messages.append({"role": "assistant", "content": f"madagascar"})
    messages.append({"role": "user", "content": f"Context: a photo of a man standing next to a bike in front of a building with a golden gate. Question: {ex_q2}."})
    messages.append({"role": "assistant", "content": f"golden gate"})
    messages.append({"role": "user", "content": f"Context: 'a photo of a group of boys playing soccer on a field. Question: {ex_q3}."})
    messages.append({"role": "assistant", "content": f"brazil"})

    context = '. '.join(all_info)
    messages.append({"role": "user", "content": f"Context: {context}. Question: {original_question}"})

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages,
      max_tokens = 10,
      temperature=temperature)

    match = re.search("[a-zA-Z].*[a-zA-Z]", response["choices"][0]['message']["content"].strip())
    if match:
      answer = match.group()
    else:
      answer = response["choices"][0]['message']["content"].strip()
    answer = answer.lower()
    
    for word in get_gpt_unknowns():
        if word in answer:
            if verbose:
               print(f"gpt {answer} for {original_question} but blip: {original_answer}")
            return original_answer

    return get_single_answer(answer)

def get_single_answer(ans):
    for f in [" or ", " and ", "/", ",", " - ", "("]:
      ans = ans.split(f)[0]
    if ans.count('"') == 2:
        ans = re.findall(r'"(.*?)"', ans)[0]
    return ans

    

def noun_gpt(original_question, temperature=0):
    """
    Helper function for prompting the GPT3 chat-based language model
    """
    messages = [
        {"role": "system", "content":  "Pick the most important noun that needs to be described to answer the question"},
        ]
    messages.append({"role": "user", "content": f"Is this a room for a boy or girl?"})
    messages.append({"role": "assistant", "content": f"room"})
    messages.append({"role": "user", "content": f"In what year was this desert first introduced?"})
    messages.append({"role": "assistant", "content": f"desert"})
    messages.append({"role": "user", "content": f"what could this gentleman be carrying in that red bag?"})
    messages.append({"role": "assistant", "content": f"bag"})
    messages.append({"role": "user", "content": f"who leaves a toilet like this?"})
    messages.append({"role": "assistant", "content": f"toilet"})
    
    
    messages.append({"role": "user", "content": f"{original_question}"})

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages,
      max_tokens = 7,
      temperature=temperature)

    match = re.search("[a-zA-Z].*[a-zA-Z]", response["choices"][0]['message']["content"].strip())
    if match:
      return match.group()
    else:
      return response["choices"][0]['message']["content"].strip()
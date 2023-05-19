import openai
import re

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
       questions = [re.search("[a-zA-Z].*", question).group() for question in questions]
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

def context_gpt(all_info, original_question, temperature=0.2):
    """
    Helper function for prompting the GPT3 chat-based language model
    """
    messages = [
        {"role": "system", "content":  "You are helpful assistant that answers a question in one or two words based on the provided context."},
        #{"role": "user", "content": f"The main question is '{original_question}'. The original answer from the visual question answering model was '{original_answer}'"}
        ]
    
    context = "\n".join(all_info)
       #context = ""
       #messages.append({"role": "user", "content": f"We asked: {question}"})
    messages.append({"role": "user", "content": f"Question: {original_question}. Context: {context}"})

    #messages.extend([{"role": "user", "content": f"Please answer the main question in one or two words: '{original_question}'. You can either repeat the original answer '{original_answer}', or improve it if necessary, still using only one or two words."}])

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
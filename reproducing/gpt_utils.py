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

def context_gpt(all_info, original_question, original_answer, temperature=0):
    """
    Helper function for prompting the GPT3 chat-based language model
    """
    messages = [
        {"role": "system", "content":  "Answer a question in one or two words by using the context"},
        ]
    
    #context = "\n".join(all_info)
    context = '. '.join(all_info)
    print('gpt context', context)
    messages.append({"role": "user", "content": f"Context: {context}. Question: {original_question}."})

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
    not_known = ["unknown", "none", "?", "information", "not specific", "no specific", "not enough", "unclear", "context", "no answer", "not provided", "not clear", "not known", "unspecified", "undetermined", "not specified", "not determined"]

    answer = answer.lower()
    
    #split_answer = answer.split()
    for word in not_known:
        if word in answer:
            print(f"gpt {answer} for {original_question} but blip: {original_answer}")
            return original_answer

    return answer
    

    

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
import openai
import re

def prompt_chat_gpt(prompt, max_tokens=64, temperature=0.7, stop=None):
  """
  Helper function for prompting the GPT3 chat-based language model
  """


  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content":  "You are a helpful assistant that generates questions that extract visual information"},
        {"role": "user", "content": f"Someone asked you '{prompt}'. List the three best distinct questions you would like to be answered \
         by a visual information retrieval system, such that you can best answer '{prompt}'. Only list questions that can be answered \
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

def summarized_gpt(questions, answers, original_question, original_answer, temperature=0.7):
    """
    Helper function for prompting the GPT3 chat-based language model
    """
    messages = [
        {"role": "system", "content":  "You are truthful assistant that gives an answer to an original question. You can utilise the information from multiple partial questions that have been answered."},
        {"role": "user", "content": f"The main question is '{original_question}'. The original answer form the visual question answering \
         model was '{original_answer}'"}
        ]
    
    for question, answer in zip(questions, answers):
       messages.append({"role": "user", "content": f"We asked: {question}"})
       messages.append({"role": "user", "content": f"The answer: {answer}"})

    messages.extend([{"role": "user", "content": f"Please answer the original question: '{original_question}'. You can either repeat the original answer '{original_answer}', or improve it if necessary. Give the shortest answer possible, in only a few words."}])

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages,
      max_tokens = 150,
      temperature=temperature)

    return re.search("[a-zA-Z].*[a-zA-Z]", response["choices"][0]['message']["content"].strip()).group()


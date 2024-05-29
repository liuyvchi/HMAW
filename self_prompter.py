import os
import pandas as pd

from openai import OpenAI
import os
from IPython.display import display, Markdown
import openai
import json
import re
from prompts import expert_prompt, ssp_prompt, on_MP_prompt

# Read OpenAI key from a file
# with open("openai_key_yuchi.txt") as f:
#     OPENAI_API_KEY = f.readline().strip()
with open("openai_key_yuchi.txt") as f:
    OPENAI_API_KEY = f.readline().strip()
openai.api_key = OPENAI_API_KEY

# Define a function to map LLM names to API names
def llm2api(llm_name):
    if llm_name=='gpt4': return  "gpt-4-1106-preview"
    elif llm_name=='gpt4t': return "gpt-4-turbo-preview"
    elif llm_name=='gpt3.5': return "gpt-3.5-turbo-0125"
    elif llm_name=='mixtral': return "mistralai/Mixtral-8x7B-Instruct-v0.1"
    elif llm_name=='mistral': return "mistralai/Mistral-7B-Instruct-v0.2"
    elif llm_name=='gpt4o': return "gpt-4o"

# Create an OpenAI agent based on the LLM name and API key
def create_agent(llm_name, openai_api_key):
    if 'gpt' in llm_name:
        agent = OpenAI(api_key=openai_api_key)
    else:
        agent = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
    return agent


import re

# Define a function to clean system messages
def clean_system_message(message):
    # Replace multiple spaces with a single space
    formatted_message = re.sub(r' +', ' ', message)
    return formatted_message

def print_dict_nicely(input_dict, indent=0):
    for key, value in input_dict.items():
        print(' ' * indent + str(key) + ':')
        if isinstance(value, dict):
            print_dict_nicely(value, indent+4)
        elif isinstance(value, list):
            for item in value:
                print(' ' * (indent+4) + "- " + str(item))
        else:
            print(' ' * (indent+4) + str(value))

# Define a function to extract the revised prompt from the LLM response
def extract_revised_prompt(response):
    # example response: **Revised Prompt:** Could you please count and tell me the number of letters 's' that appear in the word 'bananas'?
    #  but it can have mistakes as the output is from a LLM, e.g. ** might be missing both in front and back or one of those places
    # write code below to extract only the revised prompt from the response using regex
    # revised_prompt = re.search(r'\*\*Revised Prompt:\*\*(.*)', response).group(1).strip()
    revised_prompt = response.split('**New Input**:')[-1].strip()#.split('---')[0].strip()

    return revised_prompt



class llmAgent_metaP():
    def __init__(self, llm_name, openai_api_key):
        self.llm_name = llm_name
        self.agent = create_agent(llm_name, openai_api_key)

    def get_response(self, messages, max_tokens=2000, temperature=0.2, task='chat', verbose=False):

        # Client response
        client_response = self.agent.chat.completions.create(
            model=llm2api(self.llm_name),
            messages=messages,
            # max_tokens=max_tokens,
            temperature=temperature,
            seed=0,
        )
        
        output = client_response.choices[0].message.content

        if verbose:
            # Ensure there's a newline before and after the "---" for the horizontal rule
            output_ = "**Prompt**:\n{}\n\n---\n\n**Assistant**:\n{}".format(input, output)
            
            # Display the formatted text
            display(Markdown(output_))
        return [output.message.content for output in client_response.choices]

# Define a class for self-prompting
class llmAgent():
    def __init__(self, llm_name, openai_api_key):
        self.llm_name = llm_name
        self.agent = create_agent(llm_name, openai_api_key)

    def get_response(self, input, max_tokens=2000, temperature=0.2, task='chat', verbose=False):
        # Client messages
        if task == 'math':
            input = input + "\nThe last line of your response MUST be the final result and MUST MSUT be formated as '**Final Result**: <final_result>'."
        # elif task == 'codenet':
        #     input = input + "\nMust begin your response as '**Revised Code**:'"
        client_messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": input}
        ]
        
        # Client response
        client_response = self.agent.chat.completions.create(
            model=llm2api(self.llm_name),
            messages=client_messages,
            # max_tokens=max_tokens,
            temperature=temperature,
            seed=0,
        )

        output = client_response.choices[0].message.content

        if verbose:
            # Ensure there's a newline before and after the "---" for the horizontal rule
            output_ = "**Prompt**:\n{}\n\n---\n\n**Assistant**:\n{}".format(input, output)
            
            # Display the formatted text
            display(Markdown(output_))
        return output

# Define a function to get self-prompt response
def get_self_prompt_output(prompt, agent, task='chat', max_tokens=2000, temperature=0.2, verbose=False):

    # message1 = """You are given the original user input:\n 
    #     {}\n\n
        
    #     Please attach some content/text before this original user input to make up a new user input.\n 
    #     The new user input (the added user input plus the original user input) can make the AI system generate better response.\n\n

    #     The content of the added context/text is every flexible. Anything that can motivate the AI system and trigger a better response is OK.
    #     However, you should analysis the user intetion from the original user input very carefully and think about what content you should add at the begining is most appropriate.\n\n

    #     VERY IMPORTANT Notifications on the added content:\n
    #     <Start of Notifications> \n
    #     You just need to genreate the added user input.\n 
    #     The content you generate is still presented in the same tone of the original user's input.\n
    #     The added user input SHOULD NOT change the original user intention/requirement the original input.\n
    #     The added user input SHOULD NOT distort the facts in the original user input.\n
    #     The new user input (the added content plus the original user input), as a whole, should looks harmonious or coherent.\n
    #     <End of Notifications>\n\n


    #     !! YOU MUST ONLY generate the added user input.\n 

    #     """.format(prompt)
    # message1 = clean_system_message(message1)

    # response1 = agent.get_response(message1, max_tokens=max_tokens, temperature=temperature, verbose=verbose)

    # # message2 = extract_revised_prompt(response1)
    # message2 = response1 + "\n" + prompt

    # response2 = agent.get_response(message2, task=task, verbose=False)

    message1 = """You are a human user who can fairly evaluates the quality of responses to a human instruction.              
                    Given the human instruction: \n{}\n, Please give the tailored creatires to evalute the quality of the reponses to this instruction\n
                    You MUST MUST only provide the creatires.
                    """.format(prompt)
    message1 = clean_system_message(message1)
    response1 = agent.get_response(message1, max_tokens=max_tokens, temperature=temperature, verbose=verbose)

    message2 = """Here is the user input:\n
        {}.\n\n

        Here are the criterias that will be used to evalute your response:\n
        {}\n
        You just need to keep those criterias in your mind. Do not reveal the information in those criterias. Do not reveal that you have seen the criterias.\n\n

        Now, please response to the user input. 
        """.format(prompt, response1)

    message2 = clean_system_message(message2)

    response2 = agent.get_response(message2, task=task, verbose=False)

    return response1, response2


def get_expert_output(prompt, agent, max_tokens=2000, temperature=0.2, task='chat', verbose=False):
    message1 = expert_prompt.format(prompt)
    message1 = clean_system_message(message1)
    response = agent.get_response(input=message1, max_tokens=50, temperature=temperature)

    message2 = """{}\nNow given the above identity background, please answer the following instruction:\n\n{}""".format(response, prompt)
    response2 = agent.get_response(input=message2, max_tokens=max_tokens, task=task, temperature=0.2)

    return response, response2

def get_onMP_output(prompt, agent, max_tokens=2000, temperature=0.2, task='chat', verbose=False):
    message1 = on_MP_prompt.format(prompt)
    message1 = clean_system_message(message1)
    response = agent.get_response(input=message1, max_tokens=max_tokens, temperature=temperature)

    message2 = """{}\nNow given the above prompts, please answer the following instruction:\n\n{}""".format(response, prompt)
    response2 = agent.get_response(input=message2, max_tokens=max_tokens, task=task, temperature=0.2)

    return response, response2

def get_SSP_output(prompt, agent, max_tokens=2000, temperature=0.2, task='chat', verbose=False):
    message1 = ssp_prompt.format(prompt)
    message1 = clean_system_message(message1)
    response = agent.get_response(input=message1, max_tokens=max_tokens, task=task, temperature=temperature)
    personas = response.split('Final answer:')[:-1]
    final_response = response.split('Final answer:')[-1]

    return personas, final_response


def get_m2prompting_output(prompt, agent, max_tokens=2000, temperature=0.2, task='chat', verbose=False):
    message1 = """**Your ROLE**: <CEO>

        **Description**: You are the CEO of an entirely LLM-based company where all employees are LLMs. The company's goal is to generate the best possible response tailored to the user's request.

        **Company Structure**:
        CEO (LLM) -> MANAGER (LLM) -> WORKER (LLM) -> USER

        **Company Workflow**:
        1. The CEO receives the input (prompt P) from the human user.
        2. The CEO generates detailed instructions (prompt MP1) for the MANAGER LLM.
        3. According to MP1, the MANAGER then creates detailed instructions (prompt MP2) for the WORKER LLM.
        4. The WORKER LLM uses MP2 to generate the golden response (Output O) for the user.

        **IMPORTANT**:
        - As the CEO, your task is to generate the prompt MP1 for the MANAGER LLM so that the MANAGER LLM can generate golden prompt (MP2) for the WORKER LLM. The final goal is to make the output (O) of the WORKER LLM to be highly tailored, pleasing, and accurate.
        - As the CEO, do not output anything else and only provide the prompt MP1 to the MANAGER LLM. 
        - As the CEO, do not try to generate the final output for the user. This will be done by the WORKER LLM who is supervised by the MANAGER LLM.
        - If you need to repeat the human user's input, repeat it exactly without any placeholders.
        - Begin your response with **Detailed Instructions to MANAGER**:

        **Here is the input P from the human user**:
        <{}>

        Now, please generate the detailed instructions for the MANAGER LLM.
        **Detailed Isntructions to MANAGER**:""".format(prompt)

    message1 = clean_system_message(message1)
    response = agent.get_response(input=message1, max_tokens=max_tokens, temperature=temperature)

    message2 = """**Your ROLE**: <MANAGER>
        **Description**: You are the MANAGER in an entirely LLM-based company where all employees are LLMs. The company's goal is to generate the best possible response tailored to the user's request.

        **Company Structure**:
        CEO (LLM) -> MANAGER (LLM) -> WORKER (LLM) -> USER

        **Company Workflow**:
        1. The CEO receives the input (prompt P) from the human user.
        2. The CEO generates detailed instructions (prompt MP1) for the MANAGER LLM.
        3. According to MP1, the MANAGER then creates detailed instructions (prompt MP2) for the WORKER LLM.
        4. The WORKER LLM uses MP2 to generate the golden response (Output O) for the user.

        **IMPORTANT**:
        - As the MANAGER, your task is to generate the prompt MP2 for the WORKER LLM so that the WORKER LLM can provide golden response according to MP2. The final goal is to make the final output (O) of the WORKER LLM to be highly tailored, pleasing, and accurate.
        - As the MANAGER, do not output anything else. Only provide the prompts MP2 to the WORKER LLM. 
        - As the MANAGER, do not try to generate the final output for the user; this will be done by the WORKER LLM using the prompt generated by you.
        - If you need to repeat the human user's input, repeat it exactly without any placeholders.
        - Begin your response with **Detailed Instructions to WORKER**:

        **Here is the input P from the human user**:
        <{}>

        **Here is the Instructions from your CEO**:
        <{}>

        Now, please generate the detailed instructions for the WORKER LLM.
        **Detailed Isntructions to WORKER**:""".format(prompt, response)

    message2 = clean_system_message(message2)
    
    response2 = agent.get_response(input=message2, max_tokens=max_tokens, temperature=temperature)
    message3 = """**Your ROLE**: <WORKER>

        **Description**: You are the WORKER in an entirely LLM-based company where all employees are LLMs. The company's goal is to generate the best possible response tailored to the user's request.

        **Company Structure**:
        CEO (LLM) -> MANAGER (LLM) -> WORKER (LLM) -> USER

        **Company Workflow**:
        1. The CEO receives the input (prompt P) from the human user.
        2. The CEO generates detailed instructions (prompt MP1) for the MANAGER LLM.
        3. According to MP1, the MANAGER then creates detailed instructions (prompt MP2) for the WORKER LLM.
        4. The WORKER LLM uses MP2 to generate the golden response (Output O) for the user.

        **IMPORTANT**: 
        - As the WORKER, your task is to generate the final output (O) for the user with the prompt from the MANAGER. This output should be highly tailored, pleasing, and accurate.
        - Ensure the response is excellent and directly talking to the user.
        - Do not say you cannot answer.

        **Here is the input P from the human user**:
        <{}>

        **Here is the Instructions from your MANAGER**:
        <{}>
        
        Now, please generate the golden output for the user.
        **Response for the User**:""".format(prompt, response2)

    message3 = clean_system_message(message3)
    
    response3 = agent.get_response(input=message3, max_tokens=max_tokens, task=task, temperature=temperature)
    
    return response, response2, response3.split('**Response for the User**:')[-1]


# Define a class for evaluating responses
class Evaluator():
    def __init__(self, llm_name='gpt4', openai_api_key=OPENAI_API_KEY):
        self.llm_name = llm_name
        self.agent = create_agent(llm_name, openai_api_key)
    
    def get_correctness(self, instruction, answer, result, verbose=True):
        evaluator_system_message = """You are a expert to evaluates the correctness of the given result to a question. 
                                        Given the question and the corresponding corret answer for this question, you are asked to judge if the computation result from the agent is correct. 
                                        The output is in JSON format and has 2 keys: 
                                        1) "correctness": the value of this key MUST be "correct" or "incorrect". 
                                        2) "reason": briefly explain your judge in 20 words.
                                        """
        evaluation_query = "Given a problem:\n {}. \n\n This is the correct answer: \n{} \n\n This is the answer from a agent: \n{} \n\n Is the answer from the agent corret?".format(instruction, answer, result)
        
         # Use gpt4 with json format to get preference score with reasons
        client_messages = [
            {
                "role": "system",
                "content": clean_system_message(evaluator_system_message),
            },
            {
                "role": "user",
                "content": evaluation_query,
            }
        ]
        client_response = self.agent.chat.completions.create(
            model=llm2api(self.llm_name),
            messages=client_messages,
            response_format={ "type": "json_object" },
            max_tokens=1000,
            temperature=0,
            seed=0,
        )
        preference_output = client_response.choices[0].message.content
        preference_output = json.loads(preference_output)
        if verbose:
            # preference_output example output
            display(Markdown('**instruction**\n\n' + instruction))
            display(Markdown('**answer**\n\n' + answer))
            display(Markdown('**response**\n\n' + result))
            display(Markdown('**Preference Output**\n\n'))
            print_dict_nicely(preference_output)
            
        return preference_output


    def get_preference_score(self, prompt, response1, response2, task='math', verbose=True):

        # criteria_query = """You are a human user who can fairly evaluates the quality of responses to a human instruction.
        #                     Given a human instruction, your task is just to provide the criteria to evalute responses. 
        #                     The output is in JSON format and only has a key named "criterias": 
        #                     "criterias": principles or standards by which the responses may be judged or decided.
        #                     Given the human instruction: \n{}\n, Please give the creatires to evalute the quality of the reponses to this instruction
        #                     """.format(prompt)
        # criteria_query = clean_system_message(criteria_query)

        # client_messages = [
        #     # {
        #     #     "role": "system",
        #     #     "content": criteria_system_message,
        #     # },
        #     {
        #         "role": "user",
        #         "content": criteria_query,
        #     }
        # ]
        # client_response = self.agent.chat.completions.create(
        #     model=llm2api(self.llm_name),
        #     messages=client_messages,
        #     response_format={ "type": "json_object" },
        #     max_tokens=2000,
        #     temperature=0.2,
        #     seed=0,
        # )
        # preference_output = client_response.choices[0].message.content
        # preference_output = json.loads(preference_output)
        # criterias = preference_output['criterias']


        evaluation_query = """Given the human prompt: <Start of User Prompt> {} <End of User Prompt>, which of the following responses provides better user experience for the user? \n\n 
                            Response A: {} \n\n 
                            Response B: {}. \n\n 
                            
                            Your output response must be in json with the following keys: \n\n 
                            
                            1) "user-analysis": What can you infer about the user from the input user prompt? \n\n 
                            2) "pros-and-cons": Pros and Cons of Response in detail. Think step by step before responding \n\n 
                            3) "comparison": Reasoning and Comparison: use the pros and cons of each response and use the above user-analysis to indicate which response is better \n VERY VERY IMPORTANT: YOU MUST USE THE USER ANAYSIS TO CHOOSE ONE ANSWER. YOU CANT SAY THAT IT DEPENDS. YOU MUST CHOOSE).  Think step by step. Reasoning first. choice later. \n\n 
                            4) "choice": final result on which response is better. MUST be one of 'Response A', 'Response B' or 'Equal'. Must be one of 'Response A', 'Response B' or 'Equal'. YOU MUST CHOOSE.\n\n
            
                            VERY VERY IMPORTANT: I am telling you a million times: YOU MUST USE THE EXACT KEY NAMES AS ABOVE.
                            """
        evaluation_query = clean_system_message(evaluation_query).format(prompt, response1, response2)
        

        # Use gpt4 with json format to get preference score with reasons
        client_messages = [
            {
                "role": "user",
                "content": evaluation_query,
            }
        ]
        client_response = self.agent.chat.completions.create(
            model=llm2api(self.llm_name),
            messages=client_messages,
            response_format={ "type": "json_object" },
            max_tokens=1000,
            temperature=0,
            seed=0,
        )
        preference_output = client_response.choices[0].message.content
        preference_output = json.loads(preference_output)
        # preference_output['criterias'] = criterias
        if verbose:
            # preference_output example output
            display(Markdown('**Prompt**\n\n' + prompt))
            display(Markdown('**Response 1**\n\n' + response1))
            display(Markdown('**Response 2**\n\n' + response2))
            display(Markdown('**Preference Output**\n\n'))
            print_dict_nicely(preference_output)
            
        return preference_output



if __name__ == "__main__":
    # sample usage of the self-prompter
    # create agent
    agent = llmAgent('gpt3.5', OPENAI_API_KEY)
    # create evaluator
    evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)

    base_sys_prompt = "You are a helpful assistant"
    # read dataset
    df = pd.read_json('./data/general_dataset.json')
    prompt = df.iloc[3]['instruction']

    # baseline response
    response1 = agent.get_response(base_sys_prompt, prompt)
    # self_prompt response
    promptRefine_sys_promt = "You are an intelligent assistant tasked with revising user prompts. Consider how to refine the given user prompt so that the refined prompt can lead the AI model to generate a better response than the original. Do not directly provide the response to the user's prompt; just provide the refined prompts. Begin the response with **Revised Prompt**:"
    revised_prompt, response2 = get_self_prompt_response(promptRefine_sys_promt, prompt, base_sys_prompt, prompt, agent)

    # get preference score
    preference_output = evaluator.get_preference_score(prompt, response1, response2)
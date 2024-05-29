import numpy as np
import pandas as pd
import json
import os
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm

# Assuming self_prompter.py is properly set up for multiprocessing
from prompter import llmAgent, Evaluator, get_self_prompt_output, get_m2prompting_output, get_expert_output, get_SSP_output,  get_onMP_output, OPENAI_API_KEY, clean_system_message

def process_row_APE(data, llms_name):
    try:
        index, row = data
        agent = llmAgent(llms_name, OPENAI_API_KEY)
        # evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)
        
        prompt = row['context'] + "\nGenerate a response from system."

        prompt = "Have a conversation with the system using various prompts and responses to create a natural and engaging dialogue. The system should respond appropriately to the user's input and generate relevant and coherent responses to keep the conversation flowing." + prompt


        response = agent.get_response(prompt, max_tokens=4000, verbose=False)

        log = {
            'index': index,
            'prompt': prompt,
            'response': response,
        }
        return {'task_id':index, 'log': log, 'error': None}
    except Exception as e:
        print(print(f"Error processing task {index}: {e}"))
        error_log = {'task_id':index, 'log': None, 'error': str(e)}
        return error_log

def process_row_sEP(data, llms_name):
    try:
        index, row = data
        agent = llmAgent(llms_name, OPENAI_API_KEY)
        # evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)
        
        prompt = row['context'] + "\nGenerate a response from system."

        prompt = "Imaging you are an expert in the regarding field, try to answer the following instruction as professional as possible:\n\n" + prompt


        response = agent.get_response(prompt, max_tokens=4000, verbose=False)

        log = {
            'index': index,
            'prompt': prompt,
            'response': response,
        }
        return {'task_id':index, 'log': log, 'error': None}
    except Exception as e:
        print(print(f"Error processing task {index}: {e}"))
        error_log = {'task_id':index, 'log': None, 'error': str(e)}
        return error_log

def process_row_RaR(data, llms_name):
    try:
        index, row = data
        agent = llmAgent(llms_name, OPENAI_API_KEY)
        # evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)
        
        prompt = row['context'] + "\nGenerate a response from system."

        prompt = prompt + "\n\nRephrase and expand the question, and respond."


        response = agent.get_response(prompt, max_tokens=4000, verbose=False)
        response = response.split('Final answer:')[-1]

        log = {
            'index': index,
            'prompt': prompt,
            'response': response,
        }
        return {'task_id':index, 'log': log, 'error': None}
    except Exception as e:
        print(print(f"Error processing task {index}: {e}"))
        error_log = {'task_id':index, 'log': None, 'error': str(e)}
        return error_log

def process_row_onMP(data, llms_name):
    try:
        index, row = data
        agent = llmAgent(llms_name, OPENAI_API_KEY)
        # evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)
        
        prompt = row['context'] + "\nGenerate a response from system."

        prompt_gen, response = get_onMP_output(prompt, agent, max_tokens=4000, verbose=False)

        log = {
            'index': index,
            'prompt': prompt,
            'prompt_gen': prompt_gen,
            'response': response,
        }
        return {'task_id':index, 'log': log, 'error': None}
    except Exception as e:
        print(print(f"Error processing task {index}: {e}"))
        error_log = {'task_id':index, 'log': None, 'error': str(e)}
        return error_log


def process_row_SSP(data, llms_name):
    try:
        index, row = data
        agent = llmAgent(llms_name, OPENAI_API_KEY)
        # evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)
        
        prompt = row['context'] + "\nGenerate a response from system."

        personas, response = get_SSP_output(prompt, agent, max_tokens=4000, verbose=False)

        # # Get the preference score
        # preference_output = evaluator.get_preference_score(prompt, verbose=False)
        
        # preference_result = preference_output['choice']
        # result_mapped = None
        # if preference_result in ['response A', 'response B']:
        #     result_mapped = 0 if preference_result == 'response B' else 1
        
        log = {
            'index': index,
            'prompt': prompt,
            'personas': personas,
            'response': response,
            # 'criterias': preference_output['criterias'],
            # 'reason': preference_output['reason'],
            # 'preference_result': preference_result
        }
        return {'task_id':index, 'log': log, 'error': None}
    except Exception as e:
        print(print(f"Error processing task {index}: {e}"))
        error_log = {'task_id':index, 'log': None, 'error': str(e)}
        return error_log

def process_row_Expert(data, llms_name):
    try:
        index, row = data
        agent = llmAgent(llms_name, OPENAI_API_KEY)
        # evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)
        
        prompt = row['context'] + "\nGenerate a response from system."

        expert, response = get_expert_output(prompt, agent, max_tokens=4000, verbose=False)

        log = {
            'index': index,
            'prompt': prompt,
            'expert': expert,
            'response': response,
        }
        return {'task_id':index, 'log': log, 'error': None}
    except Exception as e:
        print(print(f"Error processing task {index}: {e}"))
        error_log = {'task_id':index, 'log': None, 'error': str(e)}
        return error_log

def process_row_CoT(data, llms_name):
    try:
        index, row = data
        agent = llmAgent(llms_name, OPENAI_API_KEY)
        # evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)
        
        prompt = row['context'] + "\nGenerate a response from system."

        prompt = prompt + " " + "Let's think step by step."

        response = agent.get_response(prompt, max_tokens=4000, verbose=False)

        log = {
            'index': index,
            'prompt': prompt,
            'response': response,
        }
        return {'task_id':index, 'log': log, 'error': None}
    except Exception as e:
        print(print(f"Error processing task {index}: {e}"))
        error_log = {'task_id':index, 'log': None, 'error': str(e)}
        return error_log

def process_row_base(data, llms_name):
    try:
        index, row = data
        agent = llmAgent(llms_name, OPENAI_API_KEY)
        # evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)
        
        prompt = row['context'] + "\nGenerate a response from system."

        response = agent.get_response(prompt, max_tokens=4000, verbose=False)

        log = {
            'index': index,
            'prompt': prompt,
            'response': response,
        }
        return {'task_id':index, 'log': log, 'error': None}
    except Exception as e:
        print(print(f"Error processing task {index}: {e}"))
        error_log = {'task_id':index, 'log': None, 'error': str(e)}
        return error_log

def process_row_M2P(data, llms_name):
    try:
        index, row = data
        agent = llmAgent(llms_name, OPENAI_API_KEY)
        # evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)
        
        prompt = row['context'] + "\nGenerate a response from system."

        response1, response2, response3 = get_m2prompting_output(prompt, agent, max_tokens=4000, verbose=False)

        # # Get the preference score
        # preference_output = evaluator.get_preference_score(prompt, verbose=False)
        
        # preference_result = preference_output['choice']
        # result_mapped = None
        # if preference_result in ['response A', 'response B']:
        #     result_mapped = 0 if preference_result == 'response B' else 1
        
        log = {
            'index': index,
            'prompt': prompt,
            'response1': response1,
            'response2': response2,
            'response3': response3,
            # 'criterias': preference_output['criterias'],
            # 'reason': preference_output['reason'],
            # 'preference_result': preference_result
        }
        return {'task_id':index, 'log': log, 'error': None}
    except Exception as e:
        print(print(f"Error processing task {index}: {e}"))
        error_log = {'task_id':index, 'log': None, 'error': str(e)}
        return error_log

def process_row_S2P(data, llms_name):
    try:
        index, row = data
        agent = llmAgent(llms_name, OPENAI_API_KEY)
        # evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)
        
        prompt = row['context'] + "\nGenerate a response from system."

        revised_prompt, response = get_self_prompt_output(prompt, agent, max_tokens=4000, verbose=False)

        # # Get the preference score
        # preference_output = evaluator.get_preference_score(prompt, verbose=False)
        
        # preference_result = preference_output['choice']
        # result_mapped = None
        # if preference_result in ['response A', 'response B']:
        #     result_mapped = 0 if preference_result == 'response B' else 1
        
        log = {
            'index': index,
            'prompt': prompt,
            'revised prompt': revised_prompt,
            'response': response
            # 'criterias': preference_output['criterias'],
            # 'reason': preference_output['reason'],
            # 'preference_result': preference_result
        }
        return {'task_id':index, 'log': log, 'error': None}
    except Exception as e:
        print(print(f"Error processing task {index}: {e}"))
        error_log = {'task_id':index, 'log': None, 'error': str(e)}
        return error_log


def update_progress(result, results):
    # This function will be called upon completion of each task
    results.append(result)
    pbar.update(1)

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_json('./data/FED/fed_data.json')
    # df = df[200:220]
    
    # Prepare data for multiprocessing
    data_split = [(index, row) for index, row in df.iterrows()]

    # Initialize tqdm progress bar
    global pbar
    pbar = tqdm(total=len(data_split))
    prompting = 'Expert'
    llms_name = 'mixtral'
    with Manager() as manager:
        results = manager.list()  # shared list
        num_processes = 5

        with Pool(processes=num_processes) as pool:
            for data in data_split:
                pool.apply_async(process_row_Expert, args=(data, llms_name), callback=lambda result: update_progress(result, results))
            pool.close()
            pool.join()

        # check if there is error
        for result in list(results):
            if 'error' in result and result['error'] != None:
                print(f"Error processing task {result['task_id']}: {result['error']}")
            else:
                pass
            
        results = list(results)
    # Close the progress bar
    pbar.close()
    
    # Post-process the results
    responses = [result['log'] for result in results if 'log' in result]
    errors = [result['error'] for result in results if result['error'] is not None]
    # result_mapped_list = [result['result_mapped'] for result in results if result['result_mapped'] is not None]

    #  # Compute and print the average preference score
    # if result_mapped_list:
    #     average_preference = sum(result_mapped_list) / len(result_mapped_list)
    #     print(f"Average preference score: {average_preference}")
    # else:
    #     print("No valid results to calculate an average preference score.")

    # Save detailed results and errors
    with open(f'./log2/response_fed_{prompting}_{llms_name}.json', 'w') as file:
        json.dump(responses, file, ensure_ascii=False, indent=4)

    if errors:
        with open(f'./log/errors_fed_{prompting}_{llms_name}.txt', 'w') as file:
            for e in errors:
                file.write(str(e)+'\n')
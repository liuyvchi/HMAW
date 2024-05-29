import numpy as np
import pandas as pd
import json
import os
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm

# Assuming self_prompter.py is properly set up for multiprocessing
from prompter import llmAgent, Evaluator, get_self_prompt_output, get_m2prompting_output, get_expert_output, get_SSP_output, get_onMP_output, OPENAI_API_KEY, clean_system_message

def process_row_APE(data, llms_name):
    try:
        index, row = data
        agent = llmAgent(llms_name, OPENAI_API_KEY)
        # evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)
        
        prompt = row['question']
        answer = row['answer'].split('####')[-1]

        prompt = "calculate a specific scenario or problem based on the given information.\n" + prompt


        response = agent.get_response(prompt, max_tokens=2000, task='math', verbose=False)
        result = response.split("**Final Result**:")[-1]

        log = {
            'index': index,
            'prompt': prompt,
            'answer': answer,
            'response': response,
            'result': result
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
        
        prompt = row['question']
        answer = row['answer'].split('####')[-1]

        prompt = "Imaging you are an expert in the regarding field, try to answer the following instruction as professional as possible:\n\n" + prompt +''


        response = agent.get_response(prompt, max_tokens=2000, task='math', verbose=False)
        result = response.split("**Final Result**:")[-1]

        log = {
            'index': index,
            'prompt': prompt,
            'answer': answer,
            'response': response,
            'result': result
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
        
        prompt = row['question']
        answer = row['answer'].split('####')[-1]

        prompt = prompt + "\n\nRephrase and expand the question, and respond."


        response = agent.get_response(prompt, max_tokens=2000, task='math', verbose=False)
        result = response.split("**Final Result**:")[-1]

        log = {
            'index': index,
            'prompt': prompt,
            'answer': answer,
            'response': response,
            'result': result
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
        
        prompt = row['question']
        answer = row['answer'].split('####')[-1]

        prompt_gen, response = get_onMP_output(prompt, agent, max_tokens=2000, task='math', verbose=False)
        result = response.split("**Final Result**:")[-1]

        log = {
            'index': index,
            'prompt': prompt,
            'answer': answer,
            'prompt_gen': prompt_gen,
            'response': response,
            'result': result
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
        
        prompt = row['question']
        answer = row['answer'].split('####')[-1]

        personas, response = get_SSP_output(prompt, agent, max_tokens=2000,task='math', verbose=False)
        result = response.split("**Final Result**:")[-1]

        # # Get the preference score
        # preference_output = evaluator.get_preference_score(prompt, verbose=False)
        
        # preference_result = preference_output['choice']
        # result_mapped = None
        # if preference_result in ['response A', 'response B']:
        #     result_mapped = 0 if preference_result == 'response B' else 1
        
        log = {
            'index': index,
            'prompt': prompt,
            'answer': answer,
            'personas': personas,
            'response': response,
            'result': result
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
        
        prompt = row['question']
        answer = row['answer'].split('####')[-1]

        expert, response = get_expert_output(prompt, agent, max_tokens=2000, task='math', verbose=False)
        result = response.split("**Final Result**:")[-1]

        log = {
            'index': index,
            'prompt': prompt,
            'answer': answer,
            'expert': expert,
            'response': response,
            'result': result
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
        
        prompt = row['question']
        answer = row['answer'].split('####')[-1]

        prompt = prompt + " " + "Let's think step by step."
        

        response = agent.get_response(prompt, max_tokens=2000, task='math', verbose=False)
        result = response.split("**Final Result**:")[-1]

        log = {
            'index': index,
            'prompt': prompt,
            'answer': answer,
            'response': response,
            'result': result
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
        
        prompt = row['question']
        answer = row['answer'].split('####')[-1]

        response = agent.get_response(prompt, max_tokens=2000, task='math', verbose=False)
        result = response.split("**Final Result**:")[-1]

        log = {
            'index': index,
            'prompt': prompt,
            'answer': answer,
            'response': response,
            'result': result
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
        
        prompt = row['question']
        answer = row['answer'].split('####')[-1]

        response1, response2, response3 = get_m2prompting_output(prompt, agent, max_tokens=2000, task='math', verbose=False)
        result = response3.split("**Final Result**:")[-1]

        # # Get the preference score
        # preference_output = evaluator.get_preference_score(prompt, verbose=False)
        
        # preference_result = preference_output['choice']
        # result_mapped = None
        # if preference_result in ['response A', 'response B']:
        #     result_mapped = 0 if preference_result == 'response B' else 1
        
        log = {
            'index': index,
            'prompt': prompt,
            'answer': answer,
            'response1': response1,
            'response2': response2,
            'response3': response3,
            'result': result
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
        
        prompt = row['question']
        answer = row['answer'].split('####')[-1]

        revised_prompt, response = get_self_prompt_output(prompt, agent, max_tokens=2000, task='math', verbose=False)
        result = response.split("**Final Result**:")[-1]
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
            'answer': answer,
            'response': response,
            'result': result
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
    # Initialize an empty list to hold the parsed JSON objects
    data_split = []

    # Open the JSONL file
    with open('./data/GSM8K/test.jsonl', 'r') as f:
        for i, line in enumerate(f):
            data_split.append((i, json.loads(line)))

    # data_split = data_split[:20]
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
    with open(f'./log2/response_GSM_{prompting}_{llms_name}.json', 'w') as file:
        json.dump(responses, file, ensure_ascii=False, indent=4)

    if errors:
        with open(f'./log2/errors_GSM_{prompting}_{llms_name}.txt', 'w') as file:
            for e in errors:
                file.write(str(e)+'\n')
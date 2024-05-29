import numpy as np
import pandas as pd
import json
import os
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm

# Assuming self_prompter.py is properly set up for multiprocessing
from self_prompter import llmAgent, Evaluator, get_self_prompt_output, get_m2prompting_output, OPENAI_API_KEY, clean_system_message

def process_row_compare(data):
    try:
        index, prompt, response1, response2 = data
        evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)

        # Get the preference score
        preference_output = evaluator.get_preference_score(prompt, response2, response1, verbose=False)
        
        preference_result = preference_output['choice']
        result_mapped = None
        if preference_result in ['Response A', 'Response B', 'Equal']:
            if preference_result == 'Equal':
                result_mapped = 0.5
            elif preference_result == 'Response B':
                result_mapped = 1 
            elif preference_result == 'Response A':
                result_mapped = 0  

        log = {
            'index': index,
            'prompt': prompt,
            'response A': response2,
            'response B': response1,
            'preference_result': preference_result,
            'result_mapped': result_mapped
        }
        for key, value in preference_output.items():
            log[key] = value
        return {'task_id': index, 'result_mapped': result_mapped, 'log': log, 'error': None}
    except Exception as e:
        print(print(f"Error processing task {index}: {e}"))
        error_log = {'task_id': index, 'result_mapped': None, 'log': None, 'error': str(e)}
        return error_log


def update_progress(result, results):
    # This function will be called upon completion of each task
    results.append(result)
    pbar.update(1)

if __name__ == "__main__":
    # Load the dataset
    compare = 'M2P_distort_base_gpt3.5_gpt3.5'
    dataset = 'Education'
    df1 = pd.read_json(f'./log2/response_{dataset}_M2P_distort_gpt3.5.json')
    df2 = pd.read_json(f'./log2/response_{dataset}_base_gpt3.5.json')
    
    # Prepare data for multiprocessing
    prompt_dic = {}
    for _, row in df1.iterrows():
        index_key = row['index']
        prompt_dic[index_key]=row['prompt']
    response1_dic = {}
    for _, row in df1.iterrows():
        index_key = row['index']
        response1_dic[index_key]=row['response3']
    response2_dic = {}
    for _, row in df2.iterrows():
        index_key = row['index']
        response2_dic[index_key]=row['response']
    
    
    data_split = []
    for key, prompt in prompt_dic.items():
        if key not in response1_dic.keys() or key not in response2_dic.keys():
            pass
        data_split.append((key, prompt, response1_dic[key], response2_dic[key])) 
    
    # data_split=data_split[:20]

    # Initialize tqdm progress bar
    global pbar
    pbar = tqdm(total=len(data_split))
    with Manager() as manager:
        results = manager.list()  # shared list
        num_processes = 10

        with Pool(processes=num_processes) as pool:
            for data in data_split:
                pool.apply_async(process_row_compare, args=(data,), callback=lambda result: update_progress(result, results))
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
    log_compare = [result['log'] for result in results if 'log' in result]
    errors = [result['error'] for result in results if result['error'] is not None]
    result_mapped_list = [result['result_mapped'] for result in results if result['result_mapped'] is not None]

     # Compute and print the average preference score
    if result_mapped_list:
        average_preference = sum(result_mapped_list) / len(result_mapped_list)
        print(f"Average preference score: {average_preference}")
    else:
        print("No valid results to calculate an average preference score.")

    # Save detailed results and errors
    with open(f'./log/compare_{dataset}_{compare}.json', 'w') as file:
        json.dump(log_compare, file, ensure_ascii=False, indent=4)

    if errors:
        with open(f'./log/error_{dataset}_{compare}.txt', 'w') as file:
            for e in errors:
                file.write(str(e)+'\n')
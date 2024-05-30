import numpy as np
import pandas as pd
import json
import os
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm

from prompter import llmAgent, Evaluator, get_self_prompt_output, get_m2prompting_output, OPENAI_API_KEY, clean_system_message

def process_row_compare(data):
    try:
        index, prompt, answer, result = data
        evaluator = Evaluator('gpt3.5', OPENAI_API_KEY)

        # Get the preference score
        output = evaluator.get_correctness(prompt, answer, result, verbose=False)
        
        correctness = output['correctness']
        result_mapped = None
        if correctness in ['correct', 'incorrect']:
            if correctness == 'incorrect':
                result_mapped = 0
            elif correctness == 'correct':
                result_mapped = 1 

        log = {
            'index': index,
            'prompt': prompt,
            'answer': answer,
            'result': result,
            'correctness': correctness
        }
        for key, value in output.items():
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
    setting = 'HMAW'
    dataset = 'GSM'
    df = pd.read_json(f'./log2/response_{dataset}_{setting}.json')
    
    # Prepare data for multiprocessing
    prompt_dic = {}
    answer_dic = {}
    result_dic = {}

    data_split = []
    for _, row in df.iterrows():
        data_split.append((row['index'], row['prompt'], row['answer'], row['result']))

    
    # data_split=data_split[:20]

    # Initialize tqdm progress bar
    global pbar
    pbar = tqdm(total=len(data_split))
    with Manager() as manager:
        results = manager.list()  # shared list
        num_processes = 5

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
    with open(f'./log/compare_{dataset}_{setting}.json', 'w') as file:
        json.dump(log_compare, file, ensure_ascii=False, indent=4)

    if errors:
        with open(f'./log/error_{dataset}_{setting}.txt', 'w') as file:
            for e in errors:
                file.write(str(e)+'\n')
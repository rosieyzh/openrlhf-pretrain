from math_verify import parse,verify
from argparse import ArgumentParser
from tqdm import tqdm
import jsonlines
import os
import re
import numpy as np
import torch
import re
import os
import signal
import json
import hashlib
from pathlib import Path

def reward_func(queries, prompts, answers):
    scores = []
    # batch

    for i in range(len(queries)):
        query = queries[i]
        # remove EOS token
        query = query.replace('</s>', '')
        try:
            answer = parse(str(answers[i]))
            prediction, _ = get_llm_answer(query)
            acc = verify(answer, prediction) * 1.0
        except Exception as e:
            acc = 0.0
        scores.append(acc)

    return torch.tensor(scores)


def handler(signum, frame):
    raise TimeoutError("Execution timed out!")

def execute_function(code: str, timeout=3): 
    try:
        # Set the alarm handler
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)  # Start the alarm
        local_namespace = {}
        exec(code, {}, local_namespace)
        return str(local_namespace["simple_math_problem"]())
    except TimeoutError as e:
        return None
    except Exception:
        return None
    finally:
        # Always disable the alarm after execution
        signal.alarm(0)

def execute_tinygsm_code(text):
    code = text.split('\ndef')[-1]
    code = 'def' + code
    try:
        return execute_function(code)
    except:
        return None

def execute_llm_code(text):
    try:
        # Extract code inside <llm-code> tags
        code_match = re.search(r'<llm-code>(.*?)</llm-code>', text, re.DOTALL)
        if not code_match:
            return None
        
        code = code_match.group(1).strip()
        
        # Create a dictionary for execution context
        exec_globals = {}
        
        # Split the code into lines and execute it
        lines = code.split("\n")
        last_expr = lines[-1]  # The last line of code
        timeout = 3
        
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)  # Start the alarm
            exec(code, exec_globals)
        except TimeoutError as e:

            return None
        except Exception:
            return None
        finally:
            # Always disable the alarm after execution
            signal.alarm(0)
        
        return str(eval(last_expr, exec_globals))
    except:
        return None
    
def execute_code(text):
    if '<llm-code>' in text:
        code_out = execute_llm_code(text)
        return code_out
    else:
        return execute_tinygsm_code(text)

def parse_text_answer(text):
    answer = parse(text)

def get_llm_answer(text):
    response_type = 'text'
    if '<llm-code>' in text:
        code_out = execute_llm_code(text)
        response_type = 'llm-code'
        if code_out is not None:
            return parse(code_out), 'llm-code'
    if 'def' in text:
        code_out = execute_tinygsm_code(text)
        response_type = 'tinygsm-code'
        if code_out is not None:
            return parse(code_out), 'tinygsm-code'
    
    return parse(text), response_type


def verify_llm_answer(llm_text, answer_text):
    llm_answer, _ = get_llm_answer(llm_text)
    correct_answer = parse(answer_text)
    return verify(llm_answer, correct_answer)


if __name__ == "__main__":
    gsm8k_answer = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market. #### 18"
    tinygsm_style_wrong = "\n\ndef simple_math_problem() -> int:\n    '''\n    Janet's ducks lay 16 eggs per day.\n    She eats three for breakfast every morning and bakes muffins for her friends every day with four.\n    She sells the remainder at the farmers' market daily for $2 per fresh duck egg.\n    How much in dollars does she make every day at the farmers' market?\n    '''\n    eggs_per_day = 16\n    breakfast_per_day = 3\n    muffins_per_day = 4\n    fresh_duck_eggs = 4\n    total_eggs = breakfast_per_day * 2 * 3\n    total_muffins = muffins_per_day * 2 * 3\n    total_dollars = total_eggs + total_muffins\n    dollars_per_day = total_dollars / fresh_duck_eggs\n    result = dollars_per_day\n    return result\n"
    tinygsm_style = "\n\ndef simple_math_problem() -> int:\n    '''\n    Janet\u2019s ducks lay 16 eggs per day.\n    She eats three for breakfast every morning and bakes muffins for her friends every day with four.\n    She sells the remainder at the farmers' market daily for $2 per fresh duck egg.\n    How much in dollars does she make every day at the farmers' market?\n    '''\n    eggs_per_day = 16\n    breakfast_eggs = 3\n    muffin_eggs = 4\n    remaining_eggs = eggs_per_day - breakfast_eggs - muffin_eggs\n    price_per_egg = 2\n    total_money = remaining_eggs * price_per_egg\n    result = total_money\n    return result\n"
    llm_code_style = "\n\nLet's solve this problem using Python code.\n<llm-code>\neggs_per_day = 16\neggs_per_day_for_breakfast = 3\neggs_per_day_for_muffins = 4\ndaily_earnings = eggs_per_day - eggs_per_day_for_breakfast - eggs_per_day_for_muffins\ndaily_earnings * 2\n</llm-code>\n<llm-code-output>\n96\n</llm-code-output>\nThus the farmers' market earns \\boxed{96} dollars every day."
    text_style_wrong = "\n\nTo find out how much Janet makes every day, we need to calculate the total number of eggs she uses and the total amount of money she makes from selling fresh eggs.\n\nJanet uses 3 eggs for breakfast every morning, so she uses 3 eggs for breakfast every day.\nShe eats 3 eggs for breakfast every morning, so she eats 3 eggs for breakfast every day.\nShe sells the remainder at the farmers' market daily for $2 per fresh duck egg.\n\nThe total number of eggs Janet uses is 3 (for breakfast) + 3 (for breakfast) = 6 eggs.\n\nThe total amount of money Janet makes from selling fresh eggs is 6 eggs * $2/egg = $12.\n\nSince Janet uses 3 eggs for breakfast every day, the amount of money she makes from selling fresh eggs is 3 eggs * $2/egg = $6.\n\nThe amount of money Janet makes from selling fresh eggs is the total amount of money she makes from selling eggs minus the amount of money she makes from selling fresh eggs.\n\nSo, the amount of money Janet makes every day is $12 - $6 = $6.\n\nThus, Janet makes \\boxed{6} dollars every day at the farmers' market."
    text_style = "\n\nTo find out how much Janet makes every day, we need to calculate the total number of eggs she uses and the total amount of money she makes from selling fresh eggs.\n\nJanet uses 3 eggs for breakfast every morning, so she uses 3 eggs for breakfast every day.\nShe eats 3 eggs for breakfast every morning, so she eats 3 eggs for breakfast every day.\nShe sells the remainder at the farmers' market daily for $2 per fresh duck egg.\n\nThe total number of eggs Janet uses is 3 (for breakfast) + 3 (for breakfast) = 6 eggs.\n\nThe total amount of money Janet makes from selling fresh eggs is 6 eggs * $2/egg = $12.\n\nSince Janet uses 3 eggs for breakfast every day, the amount of money she makes from selling fresh eggs is 3 eggs * $2/egg = $6.\n\nThe amount of money Janet makes from selling fresh eggs is the total amount of money she makes from selling eggs minus the amount of money she makes from selling fresh eggs.\n\nSo, the amount of money Janet makes every day is $12 - $6 = $6.\n\nThus, Janet makes \\boxed{18} dollars every day at the farmers' market."
    assert verify_llm_answer(gsm8k_answer, gsm8k_answer) == True
    assert verify_llm_answer(llm_code_style, gsm8k_answer) == True
    assert verify_llm_answer(text_style, gsm8k_answer) == True
    assert verify_llm_answer(tinygsm_style, gsm8k_answer) == True
    assert verify_llm_answer(tinygsm_style_wrong, gsm8k_answer) == False
    assert verify_llm_answer(text_style_wrong, gsm8k_answer) == False
    print('All good!')
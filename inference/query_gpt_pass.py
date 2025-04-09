import os
from argparse import ArgumentParser
import json
from collections import defaultdict
from openai import OpenAI
import re
import jsonlines
from tqdm import tqdm

def parse_answer(text: str):
    """
    Separates and returns the content inside <judge></judge> tags
    from the rest of the text.

    Parameters:
        text (str): Input string containing judge tags.

    Returns:
        Tuple[str, str]: A tuple where the first element is the content
                         inside the judge tags and the second is the rest
                         of the text.
    """
    judge_content = "".join(re.findall(r'<judge>(.*?)<\/judge>', text, re.DOTALL)).strip().split(',')
    judge_content = [x.strip() for x in judge_content] # remove white spaces
    explanation = re.sub(r'<judge>.*?<\/judge>', '', text, flags=re.DOTALL).strip()
    
    return judge_content, explanation


def build_prompt(question, answer1, answer2):
    prompt = f"""The following question was asked to an LLM like you which provided one incorrect answer before training and one correct answer after training. Tell me what changed between the 2 provided answers and why the correct one is now correct. In your answer, strictly specify from this set of possible mistakes what has happened between the 2 answers and nothing more:
    Arithmetic error – Mistakes in calculation, sign, order of operations, rounding, or undefined operations.
    Formula/application mistake – Using the wrong formula, incorrect substitutions, or misapplying rules (e.g., differentiation, integration, exponentiation, trigonometry).
    Algebraic/logic flaw – Incorrect manipulation, missing/extra terms, or flawed reasoning in problem-solving.
    Misinterpretation/misreading – Incorrect understanding of the problem, assumptions, or misusing given information.
    Notation/representation issue – Errors in variables, indexing, units, graphing, or coordinate representation.
    Incomplete answer - Incorrect solution was incomplete or collapsed (started repeating, included irrelevant content, etc.)
    
    Note that you are allowed to choose multiple errors from this list. Enclose the selected errors between <judge></judge> tags separated by commas, but put the explanations outside.

    Question: {question}
    
    Incorrect answer: {answer1}

    Correct answer: {answer2}
"""

    return prompt

def query_chatgpt(client, prompt, model="gpt-4.5-preview"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content

def load_file(file_path):
    # this is gonna be a json
    # each line is a problem 

    results = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if line:  # Check if line is not empty
                json_obj = json.loads(line)
                if 'problem' not in json_obj.keys():
                    continue
                question = json_obj['problem']
                results[question] = json_obj
    return results

def compare_and_query(client, model, baseline_results, ckpt_results, save_dir):
    questions = list(baseline_results.keys())

    writer = jsonlines.Writer(open(os.path.join(save_dir, "judge_aime_64.json"), 'w', encoding="utf-8"))
    for question in tqdm(questions):
        # check completions of each question
        baseline_q = baseline_results[question]
        ckpt_q = ckpt_results[question]
        assert len(baseline_q['completions']) >= 1

        if baseline_q['acc'] == 0.0 and ckpt_q['acc'] == 1.0:
            # if there s a change
            baseline_ans = baseline_q['completions'][0]['output']
            ckpt_ans = None
            for i in range(len(ckpt_q['completions'])):
                ckpt_ans = ckpt_q['completions'][i]['output']
                if ckpt_q['completions'][i]['acc'] == 1.0:
                    break

            prompt = build_prompt(question, baseline_ans, ckpt_ans)
            response = query_chatgpt(client, prompt, model)
            errors, explanation = parse_answer(response)

            item = {
                'problem': question,
                'baseline_answer': baseline_ans,
                'ckpt_answer': ckpt_ans,
                'gpt_errors': errors,
                'gpt_explanation': explanation
            }
            writer.write(item)
    writer.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--base', type=str)
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()
    
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    baseline_results = load_file(os.path.join(args.base, 'eval_aime3_64.json'))
    ckpt_results = load_file(os.path.join(args.ckpt, 'eval_aime3_64.json'))
    compare_and_query(client=client, model="gpt-4.5-preview", baseline_results=baseline_results, ckpt_results=ckpt_results, save_dir=args.ckpt)
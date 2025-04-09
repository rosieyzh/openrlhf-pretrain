import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from math_verify import parse, verify
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import jsonlines
import re
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from openrlhf.utils.math_verifier import verify_llm_answer, get_llm_answer

if __name__ == "__main__":
    parser = ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-c", "--checkpoint_path", type=str, help="Checkpoint path")
    parser.add_argument("-t", "--task", type=str, default='gsm8k', help="Task")
    parser.add_argument("-s", "--split", type=str, default='test', help="Split")
    parser.add_argument("-n", "--best_of_n", type=int, default=64, help="Run best-of-n inference")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("-k", "--input_key", type=str, default="question")
    parser.add_argument("-o", "--output_key", type=str, default="answer")
    parser.add_argument("--no_greedy", action="store_true", default=False, help="Don't run pass@1 with greedy decoding.")
    parser.add_argument("--no_multiple", action="store_true", default=False, help="Don't run pass@k and maj@k.")
    args = parser.parse_args()

    config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    if args.task == 'gsm8k':
        dataset = load_dataset(args.task, "main", download_config=config)
    elif args.task == 'math':
        dataset = load_dataset('HuggingFaceH4/MATH-500', download_config=config)
    elif args.task == 'aime3':
        dataset = load_dataset('AI-MO/aimo-validation-aime', download_config=config)
        args.split = 'train'
    elif args.task == 'aime':
        dataset = load_dataset('gneubig/aime-1983-2024', download_config=config)
        args.split = 'train' # Just for this
    elif args.task == 'aime2024':
        dataset = load_dataset('Maxwell-Jia/AIME_2024', download_config=config)
        args.split = 'train' # Just for this
         

    test = dataset[args.split]
    eos_token = "</s>"
    sampling_params = SamplingParams(n=args.best_of_n, temperature=args.temperature, top_p=0.95, max_tokens=2048, stop=[eos_token])
    print("Loading model ...")
    
    hf_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    examples = [doc[args.input_key] for doc in tqdm(test)]
    model = LLM(model=args.checkpoint_path)
    
    if not os.path.exists(args.checkpoint_path):
        sample_output_dir = Path("./baselines", args.checkpoint_path)
        sample_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        sample_output_dir = Path(args.checkpoint_path)

    if not args.no_greedy:
        print("Running greedy decoding...")
        greedy_output_file = os.path.join(sample_output_dir.as_posix(), f"eval_{args.task}_1.json")
        greedy_f_output = jsonlines.Writer(open(greedy_output_file, 'w', encoding="utf-8"))
        tot_length = test.num_rows
        acc_res = []
        response_type_count = defaultdict(int)
        response_type_stats = defaultdict(list)
        greedy_sampling_params = SamplingParams(n=1, temperature=0.0, top_p=0.95, max_tokens=2048, stop=[eos_token])
        results = model.generate(examples, use_tqdm=True, sampling_params=greedy_sampling_params)
        for result, doc in zip(results, test):
            prompt = result.prompt
            answer = parse(str(doc[args.output_key]))
            doc['completions'] = []
            assert len(result.outputs) == 1
            inc_output = result.outputs[0]
            generated_text = inc_output.text
            try:
                prediction, response_type = get_llm_answer(generated_text)
                acc = verify(answer, prediction) * 1.0
            except:
                acc = 0.0
                response_type = "text"
            
            response_type_count[response_type] += 1
            response_type_stats[f'{response_type}'].append(acc)
            doc['completions'].append({"output": generated_text, "acc": acc})

            if acc == 1.0:
                acc_res.append(acc)
                doc['acc'] = acc
            else:
                acc_res.append(0.0)
                doc['acc'] = 0.0
            greedy_f_output.write(doc)
        metrics = {}
        for response_type, count in response_type_count.items():
            if response_type_stats[response_type]:
                metrics[f"{response_type}_count"] = count / len(acc_res)
                metrics[f"{response_type}_acc"] = np.mean(response_type_stats[response_type])
        metrics["final_accuracy"] = np.mean(acc_res)
        print('Greedy Accuracy: {:.1%}'.format(metrics["final_accuracy"]))
        
        greedy_f_output.write(metrics)
        greedy_f_output.close()

    if not args.no_multiple:
        sample_output_file = os.path.join(sample_output_dir.as_posix(), f"eval_{args.task}_{args.best_of_n}.json")
        f_output = jsonlines.Writer(open(sample_output_file, "w", encoding="utf-8"))
        tot_length = test.num_rows
        pass_acc_res, maj_acc_res = [], []
        response_type_count = defaultdict(int)
        response_type_stats = defaultdict(list)

        results = model.generate(examples, use_tqdm=True, sampling_params=sampling_params)
        for result, doc in zip(results, test):
            prompt = result.prompt
            answer = parse(str(doc[args.output_key]))
            doc['completions'] = []

            # Pass Criterion Evaluation
            one_correct = False
            # For majority calculation
            parsed_predictions = []
            parsed_predictions_str = []
            max_elem, max_count = None, 0
            
            for inc_output in result.outputs:
                generated_text = inc_output.text
                try:
                    prediction, response_type = get_llm_answer(generated_text)
                    acc = verify(answer, prediction) * 1.0
                except:
                    acc = 0.0
                    prediction = None
                    response_type = "text"
                
                if prediction is not None:
                    if isinstance(prediction, list):
                        if len(prediction) == 0: continue
                        for pred in prediction:
                            if isinstance(pred, str):
                                parsed_predictions_str.append(pred)
                            else:
                                parsed_predictions.append(pred)
                    else:
                        parsed_predictions.append(prediction)

                response_type_count[response_type] += 1
                response_type_stats[f'{response_type}'].append(acc)
                doc['completions'].append({"output": generated_text, "acc": acc})
                
                if acc == 1.0 and not one_correct:
                    pass_acc_res.append(acc)
                    doc['acc'] = acc
                    one_correct = True
            
            if not one_correct:
                pass_acc_res.append(0.0)
                doc['acc'] = 0.0

            parser_predictions_sympy_to_str = {}
            for x in parsed_predictions:
                try:
                    parser_predictions_sympy_to_str[x] = x
                except:
                    continue
            maj_counter = Counter(list(parser_predictions_sympy_to_str.keys()))
            max_elems_with_counts = maj_counter.most_common(1)
            for max_elem, _ in max_elems_with_counts:
                try:
                    max_elem_sympy = parser_predictions_sympy_to_str[max_elem]
                    maj_acc = verify(max_elem_sympy, answer) * 1.0
                except:
                    maj_acc = 0.0

                if maj_acc == 1.0: break
            
            if maj_acc == 0.0:
                # Try with string counter
                maj_counter_str = Counter(parsed_predictions_str)
                max_elems_with_counts = maj_counter_str.most_common(1)
                for max_elem, _ in max_elems_with_counts:
                    try:
                        maj_acc = verify(max_elem, answer) * 1.0
                    except:
                        maj_acc = 0.0

                    if maj_acc == 1.0: break
            
            maj_acc_res.append(maj_acc)
            doc['maj_acc'] = maj_acc
            f_output.write(doc)
        
        metrics = {}
        for response_type, count in response_type_count.items():
            if response_type_stats[response_type]:
                metrics[f"{response_type}_count"] = count / len(pass_acc_res * args.best_of_n)
                metrics[f"{response_type}_acc"] = np.mean(response_type_stats[response_type])
        metrics["final_accuracy"] = np.mean(pass_acc_res)
        metrics["final_maj_accuracy"] = np.mean(maj_acc_res)
        
        f_output.write(metrics)
        f_output.close()
        print('Pass Accuracy: {:.1%}'.format(metrics["final_accuracy"]))
        print('Majority Accuracy: {:.1%}'.format(metrics["final_maj_accuracy"]))
        print('-' * 50)
        print('Statistics:')

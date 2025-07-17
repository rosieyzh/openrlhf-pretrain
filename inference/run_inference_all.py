import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer
import jsonlines
from pathlib import Path
from metrics import compute_greedy_metrics, compute_multiple_metrics

if __name__ == "__main__":
    parser = ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-c", "--checkpoint_path", type=str, help="Checkpoint path")
    parser.add_argument("-t", "--task", type=str, default="gsm8k", help="Task")
    parser.add_argument("-s", "--split", type=str, default="test", help="Split")
    parser.add_argument("-n", "--best_of_n", type=int, default=64, help="Run best-of-n inference")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("-k", "--input_key", type=str, default="question")
    parser.add_argument("-o", "--output_key", type=str, default="answer")
    parser.add_argument(
        "--skip_greedy", action="store_true", default=False, help="Skip pass@1 evaluation with greedy decoding"
    )
    parser.add_argument(
        "--skip_multiple", action="store_true", default=False, help="Skip pass@k and majority@k evaluation"
    )
    args = parser.parse_args()

    config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    if args.task == "gsm8k":
        dataset = load_dataset(args.task, "main", download_config=config)
    elif args.task == "math":
        dataset = load_dataset("HuggingFaceH4/MATH-500", download_config=config)

    test = dataset[args.split]
    eos_token = "</s>"
    sampling_params = SamplingParams(
        n=args.best_of_n, temperature=args.temperature, top_p=0.95, max_tokens=2048, stop=[eos_token]
    )
    print("Loading model ...")

    hf_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    examples = [doc[args.input_key] for doc in tqdm(test)]
    model = LLM(model=args.checkpoint_path)

    if not os.path.exists(args.checkpoint_path):
        sample_output_dir = Path("./baselines", args.checkpoint_path)
        sample_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        sample_output_dir = Path(args.checkpoint_path)

    if not args.skip_greedy:
        print("Running greedy decoding...")
        greedy_output_file = os.path.join(sample_output_dir.as_posix(), f"eval_{args.task}_1.json")
        greedy_f_output = jsonlines.Writer(open(greedy_output_file, "w", encoding="utf-8"))
        greedy_sampling_params = SamplingParams(n=1, temperature=0.0, top_p=0.95, max_tokens=2048, stop=[eos_token])
        results = model.generate(examples, use_tqdm=True, sampling_params=greedy_sampling_params)

        processed_data, metrics = compute_greedy_metrics(results, test, args.input_key, args.output_key)

        for doc in processed_data:
            greedy_f_output.write(doc)

        print("Greedy Accuracy: {:.1%}".format(metrics["final_accuracy"]))
        greedy_f_output.write(metrics)
        greedy_f_output.close()

    if not args.skip_multiple:
        sample_output_file = os.path.join(sample_output_dir.as_posix(), f"eval_{args.task}_{args.best_of_n}.json")
        f_output = jsonlines.Writer(open(sample_output_file, "w", encoding="utf-8"))

        results = model.generate(examples, use_tqdm=True, sampling_params=sampling_params)
        processed_data, metrics = compute_multiple_metrics(
            results, test, args.input_key, args.output_key, args.best_of_n
        )

        for doc in processed_data:
            f_output.write(doc)

        f_output.write(metrics)
        f_output.close()
        print("Pass Accuracy: {:.1%}".format(metrics["final_accuracy"]))
        print("Majority Accuracy: {:.1%}".format(metrics["final_maj_accuracy"]))

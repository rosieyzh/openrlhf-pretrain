import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import argparse
import math
import re
import numpy as np
import torch
import re
import gc
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm

from datasets import Dataset, DatasetDict
import pandas as pd
from transformers.trainer import get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer

from openrlhf.datasets import SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import SFTTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from math_verify import parse,verify
from openrlhf.utils.math_verifier import get_llm_answer

def verify_llm_answer(llm_text, answer_text):
    llm_answer, _ = get_llm_answer(llm_text)
    correct_answer = parse(answer_text)
    return verify(llm_answer, correct_answer)

def generate_ei_dataset(model, hf_tokenizer, dataset, input_key, output_key, k, temperature):
    sampling_params = SamplingParams(n=k, temperature=temperature, top_p=0.95, max_tokens=2048)
    
    # process samples ahead of time
    examples = []
    for doc in tqdm(dataset):
        if hf_tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": doc[input_key]}]
            examples.append(hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        else:
            examples.append(doc[input_key])
    
    ei_data = []
    
    results = model.generate(examples, use_tqdm=True, sampling_params=sampling_params)
    for result, doc in zip(results, dataset):
        answer = str(doc[output_key])
        for inc_output in result.outputs:
            generated_text = inc_output.text
            try:
                acc = verify_llm_answer(generated_text, answer)
            except:
                acc = 0.0
            
            if acc:
                ei_data.append(
                    {
                        input_key: doc[input_key],
                        output_key: generated_text,
                    }
                )

    # Create HF dataset and return
    ei_data_csv = pd.DataFrame(data=ei_data)
    ei_data_csv.drop_duplicates(inplace=True)
    hf_dataset = Dataset.from_pandas(ei_data_csv)
    return hf_dataset


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure save and ckpt paths - create 'ei' folder within 'args.pretrain' path
    if 'iter' not in args.pretrain:
        if os.path.exists(args.pretrain):
            output_dir = Path(args.pretrain, 'ei')
        else: # Base model
            output_dir = Path("./baselines", args.pretrain, 'ei')
    else:
        output_dir = Path(os.path.dirname(args.pretrain))

    if (args.pretrain == args.generator or "iter" not in args.generator) and 'iter' not in args.pretrain: # first iteration of EI
        args.save_path = output_dir / f"iter0{args.ei_iter_suffix}"
        args.ckpt_path = args.save_path / "ckpt"
    elif "iter" in args.generator:
        match = re.search(r'iter(\d+)', args.generator)
        ei_iter_num = int(match.group(1))
        if args.pretrain == args.generator:
            args.save_path = output_dir / f"iter{ei_iter_num + 1}{args.ei_iter_suffix}_online"
            args.ckpt_path = args.save_path / "ckpt"
        else:
            args.save_path = output_dir / f"iter{ei_iter_num + 1}{args.ei_iter_suffix}"
            args.ckpt_path = args.save_path / "ckpt"

    args.ckpt_path.mkdir(parents=True, exist_ok=True)

    # if data already exists for that iteration, just load it
    if os.path.exists(args.save_path / "data") and bool(os.listdir(args.save_path / "data")):
        print(f"Dataset already exists for current iteration. Loading from existing directory...")
        ei_dataset = DatasetDict.load_from_disk(args.save_path / "data")
        train_data = ei_dataset["train"]
        eval_data = ei_dataset["test"]
    else:
        # get expert dataset
        train_data, eval_data = blending_datasets(
            args.dataset,
            args.dataset_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            train_split=args.train_split,
            eval_split=args.eval_split,
        )
        if args.max_frac is not None:
            train_data = train_data.shuffle(seed=args.seed)
            eval_data = eval_data.shuffle(seed=args.seed)
            train_data = train_data.select(range(int(args.max_frac * len(train_data))))
            eval_data = eval_data.select(range(int(args.max_frac * len(eval_data))))
        else:
            train_data = train_data.select(range(min(args.max_samples, len(train_data))))
            eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
        
        hf_tokenizer = AutoTokenizer.from_pretrained(args.generator)
        model = LLM(model=args.generator, max_num_batched_tokens=10000)
        
        train_data = generate_ei_dataset(model, hf_tokenizer, train_data, args.input_key, args.output_key, args.k, args.temperature)
        eval_data = generate_ei_dataset(model, hf_tokenizer, eval_data, args.input_key, args.output_key, args.k, args.temperature)
        # Save dataset for reference
        new_dataset = DatasetDict({
                "train": train_data,
                "test": eval_data,
        })
        
        new_dataset.save_to_disk(args.save_path / "data")
        print(f"Saved new dataset to {args.save_path / 'data'}")

        # Delete the llm object and free the memory
        destroy_model_parallel()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()

    # configure model
    # load huggingface model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
        use_liger_kernel=args.use_liger_kernel,
    )
    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # prepare for data and dataset
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
        multiturn=args.multiturn,
    )
    eval_dataset = SFTDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
        multiturn=args.multiturn,
    )

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,
        False,
        eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # prepare models
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    # configure Trainer
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--universal_ckpt", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--torch_compile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # EI sampling args
    parser.add_argument("--k", type=int, default=100, help="number of samples used for creating expert dataset.")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature used for creating expert dataset.")
    parser.add_argument("--generator", type=str, default=None)
    parser.add_argument("--ei_iter_suffix", type=str, default='')

    # SFT
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # ring-attention
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # packing SFT samples without CrossAttention
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")
    parser.add_argument("--multiturn", action="store_true", default=False, help="Use compacted multiturn dataset")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_frac", type=float, default=None, help="Max fraction of data to train on")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ei")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ei_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.multiturn:
        assert args.apply_chat_template, "apply_chat_template must be enabled when using multiturn format"

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    # TODO: [packing samples]
    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    train(args)

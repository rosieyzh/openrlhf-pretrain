#!/bin/bash

# Define base directory
base_dir="<PATH_TO_CHECKPOINT>"

# Define input and output directory names
input_name=$1 #"OLMo-150M-as_fm3_tg_omi2/latest-unsharded"
output_name="$input_name-hf"

# Define tokenizer name
tokenizer_name="meta-llama/Llama-2-7b-hf"

# Construct full paths
input_dir="$base_dir/$input_name"
output_dir="$base_dir/$output_name"

echo "Input directory: $input_dir"
echo "Output directory: $output_dir"

# Run the command to convert weights to HF format
script='<PATH_TO_OLMO>/convert_olmo_weights_to_hf.py'
python $script --input_dir $input_dir --output_dir $output_dir --no_fix_eos_token_id

# Save tokenizer
save_tokenizer_script=save_tokenizer.py
python $save_tokenizer_script --tokenizer_name $tokenizer_name --target_dir $output_dir

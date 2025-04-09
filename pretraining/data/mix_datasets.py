import os
import random

# Base directory
base_dir = "<PATH_TO_DATA>"

# List of directories with their corresponding probabilities
directories = {
    f"{base_dir}/algebraic-stack-tokenized": 1.0,
    f"{base_dir}/finemath3-tokenized": 1.0,
    f"{base_dir}/openmathinstruct2-tokenized": 1.0,
    f"{base_dir}/MetaMathQA-tokenized": 1.0
}

# New directory
new_directory = f"{base_dir}/as_fm3_omi2_mmqa"
os.makedirs(new_directory, exist_ok=True)

# Iterate over directories and create symbolic links
for dir_path, prob in directories.items():
    if not os.path.exists(dir_path):
        print(f"Skipping {dir_path}: Directory does not exist.")
        continue

    if prob == 0.0:
        print(f"Skipping {dir_path}: Probability set to 0.")
        continue

    # Maintain directory structure inside new_directory
    relative_dir = os.path.relpath(dir_path, base_dir)
    target_dir = os.path.join(new_directory, relative_dir)
    os.makedirs(target_dir, exist_ok=True)

    # Get all base filenames (without .index or .metadata) while filtering out "unshuffled"
    base_files = set()
    for file in os.listdir(dir_path):
        if "unshuffled" not in file and file.endswith(".ds"):
            base_name = file.rsplit(".ds", 1)[0]  # Extract the prefix (e.g., "00009_00000_shuffled")
            base_files.add(base_name)

    if not base_files:
        print(f"Skipping {dir_path}: No valid files found.")
        continue

    # Select files based on probability
    if prob >= 1.0:
        selected_bases = base_files
    elif prob < 1.0 and prob > 0.0:
        selected_bases = random.sample(base_files, int(len(base_files) * prob))
    elif prob == 0.0:
        continue

    # Link files, preserving the directory structure
    for base in selected_bases:
        num_repeats = 1 if prob <= 1.0 else int(prob)
        for repeat in range(num_repeats):
            for ext in [".ds", ".ds.index", ".ds.metadata"]:
                src = os.path.join(dir_path, base + ext)
                rep_str = '' if repeat == 0 else f"_{repeat}"
                dest = os.path.join(target_dir, base + rep_str + ext)

                if os.path.exists(src):  # Ensure the source file exists before linking
                    if not os.path.exists(dest):  # Avoid overwriting existing links
                        os.symlink(src, dest)

print("Symbolic links created successfully, maintaining directory structure.")
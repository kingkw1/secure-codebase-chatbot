import os
from datasets import load_dataset

parent_folder = "evaluation_datasets"
if not os.path.exists(parent_folder):
    os.makedirs(parent_folder)

storage_folder = os.path.join(parent_folder, "codesearchnet_python")
if os.path.exists(storage_folder):
    print(f"Dataset already exists at {storage_folder}.")
else:
    dataset = load_dataset("code_search_net", "python", trust_remote_code=True)
    dataset.save_to_disk(storage_folder)

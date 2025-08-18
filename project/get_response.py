import os
import subprocess
import prompt
import datetime
from tqdm import tqdm

RESPONSE_DIR = "response"

def get_installed_models():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    models = []
    for line in result.stdout.strip().splitlines():
        if not line.strip() or line.lower().startswith("model") or line.lower().startswith("name"):
            continue
        model_name = line.split()[0]
        models.append(model_name) 
    return models

def query_model(model, prompt_text):
    result = subprocess.run(
        ["ollama", "run", model, prompt_text],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def get_prompt_choices():
    prompt_choices = []
    for attr in dir(prompt):
        if attr.isupper() and "PROMPT" in attr:
            prompt_choices.append((attr, getattr(prompt, attr)))
    return prompt_choices

def select_prompts(prompt_choices):
    print("Available prompts:")
    for idx, (name, _) in enumerate(prompt_choices, 1):
        print(f"{idx}. {name}")
    prompt_selection = input(
        "\nEnter prompt numbers separated by commas (e.g., 1,2 for multiple): "
    ).strip()
    try:
        prompt_indices = set(int(i.strip()) - 1 for i in prompt_selection.split(",") if i.strip())
        selected_prompts = [(prompt_choices[i][0], prompt_choices[i][1]) for i in prompt_indices if 0 <= i < len(prompt_choices)]
    except Exception:
        print("Invalid prompt selection. Exiting.")
        return []
    if not selected_prompts:
        print("No valid prompts selected. Exiting.")
        return []
    return selected_prompts

def fill_prompt_json(selected_prompts):
    prompt_texts = []
    objects_dir = os.path.join(os.getcwd(), "objects")
    subdirs = []
    for root, dirs, files in os.walk(objects_dir):
        if root == objects_dir:
            subdirs = dirs
            break

    for prompt_name, prompt_text in selected_prompts:
        if "{objects_json}" in prompt_text:
            if not subdirs:
                print("No subdirectories found under the 'objects' directory.")
                continue
            print(f"Prompt '{prompt_name}' requires a JSON file with objects and coordinates.")
            print("Available folders under 'objects/':")
            for idx, folder in enumerate(subdirs, 1):
                print(f"{idx}. {folder}")
            folder_idx = input("Enter the number of the folder to use: ").strip()
            try:
                folder_idx = int(folder_idx) - 1
                if 0 <= folder_idx < len(subdirs):
                    selected_folder = os.path.join(objects_dir, subdirs[folder_idx])
                else:
                    print("Invalid folder selection. Skipping this prompt.")
                    continue
            except Exception:
                print("Invalid folder selection. Skipping this prompt.")
                continue

            json_files = []
            for root, _, files in os.walk(selected_folder):
                for file in files:
                    if file.endswith(".json"):
                        rel_path = os.path.relpath(os.path.join(root, file), os.getcwd())
                        json_files.append(rel_path)
            if not json_files:
                print("No JSON files found in the selected folder.")
                continue
            print("Available JSON files:")
            for idx, path in enumerate(json_files, 1):
                print(f"{idx}. {path}")
            file_idx = input("Enter the number of the JSON file to use: ").strip()
            try:
                file_idx = int(file_idx) - 1
                if 0 <= file_idx < len(json_files):
                    json_path = json_files[file_idx]
                else:
                    print("Invalid file selection. Skipping this prompt.")
                    continue
                with open(json_path, "r") as f:
                    objects_json = f.read()
                prompt_text = prompt_text.format(objects_json=objects_json)
            except Exception as e:
                print(f"Failed to read JSON file: {e}")
                continue
        prompt_texts.append((prompt_name, prompt_text))
    return prompt_texts

def select_models(models):
    print("\nAvailable models:")
    for idx, model in enumerate(models, 1):
        print(f"{idx}. {model}")
    selection = input(
        "\nEnter model numbers separated by commas (e.g., 1,3 for multiple): "
    ).strip()
    try:
        indices = set(int(i.strip()) - 1 for i in selection.split(",") if i.strip())
        selected_models = [models[i] for i in indices if 0 <= i < len(models)]
    except Exception:
        print("Invalid model selection. Exiting.")
        return []
    if not selected_models:
        print("No valid models selected. Exiting.")
        return []
    return selected_models

def write_responses(prompt_texts, selected_models):
    for prompt_name, prompt_text in prompt_texts:
        safe_prompt_name = prompt_name.replace(":", "_").replace("/", "_")
        json_file_name = "nojson"
        if "{objects_json}" not in prompt_text:
            import re
            match = re.search(r'"([^"]+\.json)"', prompt_text)
            if match:
                json_file_name = os.path.basename(match.group(1)).replace("/", "_")
        else:
            json_file_name = "nojson"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(selected_models) == 1:
            safe_model_name = selected_models[0].replace(":", "_").replace("/", "_")
            file_name = f"{safe_prompt_name}_{json_file_name}_{safe_model_name}_{timestamp}.txt"
        else:
            joined_names = "_".join(m.replace(":", "_").replace("/", "_") for m in selected_models)
            file_name = f"{safe_prompt_name}_{json_file_name}_{joined_names}_{timestamp}.txt"
        all_responses_path = os.path.join(RESPONSE_DIR, file_name)

        with open(all_responses_path, "w") as f:
            f.write(f"Prompt ({prompt_name}):\n\n")
            for model in tqdm(selected_models, desc=f"Querying models for '{prompt_name}'"):
                print(f"Querying {model} with {prompt_name}...")
                response = query_model(model, prompt_text)
                f.write(f"=== {model} ===\n")
                f.write(response + "\n\n")
        print(f"Responses for prompt '{prompt_name}' saved in '{all_responses_path}'.")
def main():
    os.makedirs(RESPONSE_DIR, exist_ok=True)
    prompt_choices = get_prompt_choices()
    if not prompt_choices:
        print("No prompts found in prompt.py.")
        return

    selected_prompts = select_prompts(prompt_choices)
    if not selected_prompts:
        return

    prompt_texts = fill_prompt_json(selected_prompts)
    if not prompt_texts:
        print("No valid prompts after JSON insertion. Exiting.")
        return

    models = get_installed_models()
    if not models:
        print("No models found.")
        return

    selected_models = select_models(models)
    if not selected_models:
        return

    write_responses(prompt_texts, selected_models)

if __name__ == "__main__":
    main()
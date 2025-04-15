import os
import json
import re

def create_solution_folder(base_dir, question_id, question_text):
    """
    Create a folder for storing the solution steps and final answer.
    Returns the path to the created folder.
    """
    # Clean question text for safe folder name
    safe_question = re.sub(r'[^a-zA-Z0-9_]+', '_', question_text[:50])
    folder_name = f"{question_id:03d}_{safe_question}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_step_json(folder_path, step_no, step_data):
    """
    Save a step as a JSON file.
    """
    step_file = os.path.join(folder_path, f"step_{step_no:02d}.json")
    with open(step_file, "w") as f:
        json.dump(step_data, f, indent=2)

def save_final_answer(folder_path, answer_text):
    """
    Save the final answer as a text file.
    """
    final_file = os.path.join(folder_path, "final_answer.txt")
    with open(final_file, "w") as f:
        f.write(answer_text)
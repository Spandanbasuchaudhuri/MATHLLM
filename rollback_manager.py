import os

def check_and_rollback(folder_path: str, current_step: int):
    """
    Super basic rollback: if a step fails, remove the step_X.json file.
    """
    step_file = os.path.join(folder_path, f"step_{current_step:02d}.json")
    if os.path.exists(step_file):
        os.remove(step_file)
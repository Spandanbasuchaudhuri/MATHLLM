import time
import re

def generate_question_id():
    """
    Use the Unix time as a unique ID.
    """
    return int(time.time())

def parse_steps_from_text(full_text: str):
    """
    Naive parser that tries to find lines starting with 'Step X:' and 'Final Answer:'.
    Returns: steps (list of str), final_answer (str or None)
    """
    # Split around "Final Answer:"
    parts = full_text.split("Final Answer:")
    if len(parts) == 1:
        text_for_steps = parts[0]
        final_answer = ""
    else:
        text_for_steps = parts[0]
        final_answer = parts[1].strip()
        
    # Now find each "Step X:" in text_for_steps
    pattern = r"(Step\s*\d+:)(.*?)(?=Step\s*\d+:|$)"
    matches = re.findall(pattern, text_for_steps, flags=re.DOTALL)
    
    # Prevent duplicates
    steps = []
    seen_steps = set()
    for label, content in matches:
        step_num = int(re.search(r'\d+', label).group(0))
        if step_num in seen_steps:
            continue
        seen_steps.add(step_num)
        
        step_str = (label + content).strip()
        steps.append(step_str)
    
    return steps, final_answer
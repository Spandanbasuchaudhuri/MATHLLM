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

    Returns:
      steps (list of str), final_answer (str or None)
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
    # We'll do a simple regex approach
    # Example lines might look like:
    # Step 1: blah blah
    # Step 2: more stuff
    # ...
    pattern = r"(Step\s*\d+:)(.*?)(?=Step\s*\d+:|$)"
    matches = re.findall(pattern, text_for_steps, flags=re.DOTALL)

    # Each match is (step_label, step_content)
    steps = []
    for label, content in matches:
        step_str = (label + content).strip()
        steps.append(step_str)

    return steps, final_answer
import streamlit as st
import os
import re
from llama_model import query_llama3_stream
from sympy_solver import sympy_check
from utils import generate_question_id
from step_logger import create_solution_folder, save_step_json, save_final_answer
from rollback_manager import check_and_rollback

# Ensure the storage directory exists
if not os.path.exists("math_solutions"):
    os.makedirs("math_solutions")

st.set_page_config(page_title="Clean Math Solver", layout="centered")
st.title("🧮 Clean Math Solver")

question = st.text_area("Enter your math question:", value="Find the derivative of x^3 * sin(x).")

def clean_text(text):
    """
    Clean text by removing extraneous asterisks and extra whitespace.
    """
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_steps_and_answer(text):
    """
    Parse and clean individual steps and the final answer from the model's response.
    """
    # Extract final answer using a regex search.
    final_answer_match = re.search(r"Final Answer:?(.*?)$", text, re.DOTALL)
    final_answer = final_answer_match.group(1).strip() if final_answer_match else None
    final_answer = clean_text(final_answer) if final_answer else None

    # Limit the text for steps to before the final answer.
    text_for_steps = text[:final_answer_match.start()] if final_answer_match else text

    # Define a regex pattern to capture steps.
    step_pattern = r"Step\s*(\d+):?\s*(.*?)(?=Step\s*\d+:|Final Answer:|$)"
    step_matches = re.finditer(step_pattern, text_for_steps, re.DOTALL)
    
    steps = []
    for match in step_matches:
        step_num = int(match.group(1))
        step_text = clean_text(match.group(2))
        steps.append({'number': step_num, 'content': step_text})
    steps.sort(key=lambda x: x['number'])
    return steps, final_answer

if st.button("🔍 Solve Question"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Solving..."):
        # Generate a unique ID and create a folder to log the solution.
        question_id = generate_question_id()
        folder_path = create_solution_folder("math_solutions", question_id, question)

        prompt = f"""
You are a math tutor. Solve this problem step-by-step.
Use 'Step 1:', 'Step 2:', etc.
After all steps, write 'Final Answer:' on a new line.

Problem: {question}
"""
        response = ""
        for chunk in query_llama3_stream(prompt):
            response += chunk

        steps, final_answer = parse_steps_and_answer(response)

        # Display each step with bullet formatting.
        for step in steps:
            step_num = step["number"]
            step_text = step["content"]
            is_valid = sympy_check(f"Step {step_num}: {step_text}")
            status = "✅" if is_valid else "❌"
            st.markdown(f"- **Step {step_num} {status}:**")
            st.markdown(f"  - {step_text}")
            save_step_json(folder_path, step_num, {
                "step_no": step_num,
                "model_output": step_text,
                "rollback_triggered": not is_valid
            })
            if not is_valid:
                check_and_rollback(folder_path, step_num)

        # Display the final answer with separation and LaTeX rendering.
        if final_answer:
            st.markdown("---")
            st.markdown("### ✅ Final Answer")
            st.latex(final_answer)
            save_final_answer(folder_path, final_answer)
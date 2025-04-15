import streamlit as st
import os
import re
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from llama_model import query_llama3_stream
from sympy_solver import sympy_check
from utils import generate_question_id
from step_logger import create_solution_folder, save_step_json, save_final_answer
from rollback_manager import check_and_rollback

# Load TrOCR model and processor for OCR of math images
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

def extract_math_from_image(image: Image.Image) -> str:
    pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def clean_text(text):
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_steps_and_answer(text):
    final_answer_match = re.search(r"Final Answer:?(.*?)$", text, re.DOTALL)
    final_answer = final_answer_match.group(1).strip() if final_answer_match else None
    final_answer = clean_text(final_answer) if final_answer else None
    text_for_steps = text[:final_answer_match.start()] if final_answer_match else text
    step_pattern = r"Step\s*(\d+):?\s*(.*?)(?=Step\s*\d+:|Final Answer:|$)"
    step_matches = re.finditer(step_pattern, text_for_steps, re.DOTALL)
    steps = []
    seen = set()
    for match in step_matches:
        step_num = int(match.group(1))
        if step_num in seen:
            continue  # Skip duplicate step numbers
        seen.add(step_num)
        step_text = clean_text(match.group(2))
        steps.append({'number': step_num, 'content': step_text})
    steps.sort(key=lambda x: x['number'])
    return steps, final_answer

if not os.path.exists("math_solutions"):
    os.makedirs("math_solutions")

st.set_page_config(page_title="Clean Math Solver", layout="centered")
st.title("🧮 Clean Math Solver")

input_mode = st.radio("Select Input Mode", ["Text", "Image"])
question = ""

if input_mode == "Text":
    question = st.text_area("Enter your math question:", value="Find the derivative of x^3 * sin(x).")
elif input_mode == "Image":
    uploaded_file = st.file_uploader("Upload an image of a math problem", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        ocr_text = extract_math_from_image(image)
        st.markdown("### OCR Extracted Text")
        st.text_area("Editable OCR Text", value=ocr_text, height=150)
        question = st.text_area("Math Question from OCR (editable)", value=ocr_text)

if st.button("🔍 Solve Question"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Solving..."):
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
        for step in steps:
            step_num = step["number"]
            step_text = step["content"]
            is_valid = sympy_check(f"Step {step_num}: {step_text}")
            status = "✅" if is_valid else "❌"
            with st.expander(f"Step {step_num} {status}"):
                # Try to render as LaTeX if possible, else as text
                latex_match = re.search(r"\$([^\$]+)\$", step_text)
                if latex_match:
                    st.latex(latex_match.group(1))
                    st.write(step_text.replace(latex_match.group(0), ""))
                else:
                    st.write(step_text)
            save_step_json(folder_path, step_num, {
                "step_no": step_num,
                "model_output": step_text,
                "rollback_triggered": not is_valid
            })
            if not is_valid:
                check_and_rollback(folder_path, step_num)

        if final_answer:
            st.markdown("---")
            st.success("### Final Answer")
            # Remove any trailing text like "That’s sit!"
            clean_final = re.sub(r"That.?s sit!?$", "", final_answer, flags=re.IGNORECASE).strip()
            st.latex(clean_final)
            save_final_answer(folder_path, clean_final)
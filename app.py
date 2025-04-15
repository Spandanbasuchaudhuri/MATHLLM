import streamlit as st
import os
import re
import shutil
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from llama_model import query_llama3_stream
from sympy_solver import sympy_check, parse_math_expression
from utils import generate_question_id
from step_logger import create_solution_folder, save_step_json, save_final_answer
from rollback_manager import check_and_rollback

# Load TrOCR model and processor for OCR of math images
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

def extract_math_from_image(image: Image.Image) -> str:
    """Extract math text from an image using the TrOCR model."""
    pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def clean_text(text):
    """Clean text by removing extraneous asterisks and extra whitespace."""
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_steps_and_answer(text):
    """Parse and clean individual steps and the final answer from the model's response."""
    final_answer_match = re.search(r"Final Answer:?(.*?)$", text, re.DOTALL)
    final_answer = final_answer_match.group(1).strip() if final_answer_match else None
    final_answer = clean_text(final_answer) if final_answer else None
    text_for_steps = text[:final_answer_match.start()] if final_answer_match else text
    step_pattern = r"Step\s*(\d+):?\s*(.*?)(?=Step\s*\d+:|Final Answer:|$)"
    step_matches = re.finditer(step_pattern, text_for_steps, re.DOTALL)
    steps = []
    for match in step_matches:
        step_num = int(match.group(1))
        step_text = clean_text(match.group(2))
        steps.append({'number': step_num, 'content': step_text})
    steps.sort(key=lambda x: x['number'])
    return steps, final_answer

def zip_solution_folder(folder_path):
    zip_path = folder_path + ".zip"
    shutil.make_archive(folder_path, 'zip', folder_path)
    return zip_path

# Ensure the storage directory exists
if not os.path.exists("math_solutions"):
    os.makedirs("math_solutions")

st.set_page_config(page_title="Clean Math Solver", layout="centered")
st.title("🧮 Clean Math Solver")

# Allow user to choose input mode: Text or Image
input_mode = st.radio("Select Input Mode", ["Text", "Image"])
question = ""

if input_mode == "Text":
    question = st.text_area("Enter your math question:", value="Find the derivative of x^3 * sin(x).")
elif input_mode == "Image":
    uploaded_file = st.file_uploader("Upload an image of a math problem", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        try:
            ocr_text = extract_math_from_image(image)
        except Exception as e:
            st.error(f"OCR failed: {e}")
            st.stop()
        st.markdown("### OCR Extracted Text")
        st.text_area("Editable OCR Text", value=ocr_text, height=150)
        question = st.text_area("Math Question from OCR (editable)", value=ocr_text)

if st.button("🔍 Solve Question"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Solving..."):
        # Generate a unique question ID and create a logging folder.
        question_id = generate_question_id()
        folder_path = create_solution_folder("math_solutions", question_id, question)

        prompt = f"""
You are a math tutor. Solve this problem step-by-step.
Use 'Step 1:', 'Step 2:', etc.
After all steps, write 'Final Answer:' on a new line.
Problem: {question}
"""
        response = ""
        try:
            for chunk in query_llama3_stream(prompt):
                response += chunk
        except Exception as e:
            st.error(f"LLM failed: {e}")
            st.stop()

        steps, final_answer = parse_steps_and_answer(response)
        all_valid = True
        for step in steps:
            step_num = step["number"]
            step_text = step["content"]
            is_valid, validation_msg = sympy_check(step_text)
            status = "✅" if is_valid else "❌"
            st.markdown(f"- **Step {step_num} {status}:**")
            st.markdown(f" - {step_text}")
            if not is_valid:
                st.error(f"Step {step_num} failed validation: {validation_msg}")
                check_and_rollback(folder_path, step_num)
                all_valid = False
            save_step_json(folder_path, step_num, {
                "step_no": step_num,
                "model_output": step_text,
                "rollback_triggered": not is_valid,
                "validation_message": validation_msg
            })

        if final_answer:
            st.markdown("---")
            st.markdown("### ✅ Final Answer")
            try:
                st.latex(final_answer)
            except Exception:
                st.write(final_answer)
            save_final_answer(folder_path, final_answer)

        # Download solution as ZIP
        if os.path.exists(folder_path):
            zip_path = zip_solution_folder(folder_path)
            with open(zip_path, "rb") as f:
                st.download_button("Download Solution Folder (ZIP)", f, file_name=os.path.basename(zip_path))

        if all_valid:
            st.success("All steps validated successfully!")
        else:
            st.warning("Some steps failed validation. Please review the steps above.")
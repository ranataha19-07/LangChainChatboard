import io
import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
import pytesseract
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pypdf import PdfReader
from docx import Document

load_dotenv()


def extract_text_from_file(uploaded_file):
    """Extract plain text from uploaded PDF, DOCX, TXT, Excel, or Image files."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    
    # --- Excel files extraction ---
    elif name.endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(io.BytesIO(data))
        return df.to_string()
    
    # --- Image files extraction (OCR) ---
    elif name.endswith((".png", ".jpg", ".jpeg")):
        try:
            img = Image.open(io.BytesIO(data))
            text = pytesseract.image_to_string(img)
            return text if text.strip() else "[Image content could not be read as text]"
        except Exception as e:
            return f"[Error processing image: {e}]"
            
    else:  # .txt or fallback
        return data.decode("utf-8", errors="ignore")


# 1. Page Configuration
st.set_page_config(
    page_title="HireLens AI",
    page_icon="🎯",
    layout="centered"
)

# 2. Initialize Session State Variables
if "api_authenticated" not in st.session_state:
    st.session_state["api_authenticated"] = False
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 3. Lock Screen: Ask for API Key First
if not st.session_state["api_authenticated"]:
    st.title("🔐 HireLens AI - Access Portal")
    st.write("Please enter your OpenAI API Key to unlock the smart recruitment dashboard.")

    # API Key Input Form
    api_key_input = st.text_input("OpenAI API Key", type="password")

    if st.button("Unlock Dashboard"):
        if api_key_input.strip() != "":
            st.session_state["api_key"] = api_key_input.strip()
            st.session_state["api_authenticated"] = True
            st.rerun()  # Refresh app to load the main interface
        else:
            st.error("Please enter a valid API key to proceed.")

    # Stop execution here so the dashboard doesn't load until unlocked
    st.stop()

# ==========================================
# 4. MAIN INTERFACE (Loaded after unlock)
# ==========================================
st.title("🎯 HireLens AI")
st.caption("Transform raw candidate CVs, portfolios, and hiring spreadsheets into structured evaluation metrics, fit analytics, and smart interview cues.")

# Define the expert system prompt role for the HR recruiting analyst
system_prompt = (
    "You are an expert talent evaluation specialist and senior HR recruiter. "
    "When a user provides a candidate resume, portfolio, profile, or data sheet, review it thoroughly and generate a clean, structured analysis covering: "
    "1. Candidate Fit Metric (out of 10). "
    "2. Top 3 core strengths (technical expertise, domain experience, key milestones). "
    "3. Top 3 potential gaps, risks, or inconsistencies. "
    "4. A definitive hiring recommendation (e.g., proceed to technical round, explore further, or pass). "
    "Maintain a professional tone using structured markdown formatting and lists."
)

# Ensure System Message is tracked in session history
if not any(isinstance(x, SystemMessage) for x in st.session_state["messages"]):
    st.session_state["messages"].append(SystemMessage(content=system_prompt))

# Display prior chat messages/reports
for msg in st.session_state["messages"][1:]:  # Skip showing the raw system message
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Dedicated File Uploader widget (No typing required to upload & evaluate)
uploaded_file = st.file_uploader(
    "📁 Upload Candidate Document (PDF, DOCX, TXT, Excel, Image)",
    type=["pdf", "docx", "txt", "xlsx", "xls", "xlsm", "png", "jpg", "jpeg"]
)

if uploaded_file:
    # Check if this specific file was already processed to avoid duplicate triggers on rerun
    file_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
    if file_key not in st.session_state:
        extracted_text = extract_text_from_file(uploaded_file)
        full_prompt = f"--- Document Source: {uploaded_file.name} ---\n{extracted_text}"

        # Append to message history
        st.session_state["messages"].append(HumanMessage(content=full_prompt))
        
        with st.chat_message("user"):
            st.markdown(f"📎 **Uploaded Document:** {uploaded_file.name}")

        try:
            chat_model = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.3,
                openai_api_key=st.session_state["api_key"]
            )

            with st.chat_message("assistant"):
                with st.spinner("Processing candidate details..."):
                    response = chat_model.invoke(st.session_state["messages"])
                    st.markdown(response.content)
                    st.session_state["messages"].append(AIMessage(content=response.content))
            
            st.session_state[file_key] = True
        except Exception as e:
            st.error(f"An error occurred while communicating with the OpenAI API: {e}")

# Optional chat input for follow-up questions after file analysis
user_chat = st.chat_input("Ask follow-up questions about the candidate...")
if user_chat:
    st.session_state["messages"].append(HumanMessage(content=user_chat))
    with st.chat_message("user"):
        st.markdown(user_chat)

    try:
        chat_model = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=st.session_state["api_key"]
        )
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_model.invoke(st.session_state["messages"])
                st.markdown(response.content)
                st.session_state["messages"].append(AIMessage(content=response.content))
    except Exception as e:
        st.error(f"An error occurred: {e}")
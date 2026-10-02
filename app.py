import os
import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf
import docx

# ----------------- CONFIG & INITIALIZATION -----------------
st.set_page_config(page_title="Multimodal Personal AI", page_icon="🤖", layout="wide")

API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("Missing GEMINI_API_KEY. Please configure it in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-flash-latest")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_content" not in st.session_state:
    st.session_state.doc_content = ""
if "created_doc" not in st.session_state:
    st.session_state.created_doc = ""

# ----------------- HELPER FUNCTIONS -----------------
def extract_text_from_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.name.endswith(".pdf"):
            reader = pypdf.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
    return text

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.title("⚙️ Assistant Controls")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.doc_content = ""
        st.session_state.created_doc = ""
        st.rerun()

    # Module F: Conversation Modes
    mode = st.selectbox(
        "Chatbot Mode:",
        ["General Assistant", "Document RAG Intelligence", "Empathetic Emotional Support"]
    )
    
    st.divider()
    
    # Module A: Multimodal Uploads
    st.subheader("📎 Attachments")
    uploaded_image = st.file_uploader("Upload Image (Vision)", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_image:
        st.image(uploaded_image, caption="Loaded Image Preview", use_container_width=True)

    uploaded_doc = st.file_uploader("Upload Document (RAG)", type=["pdf", "docx", "txt"])
    if uploaded_doc:
        extracted = extract_text_from_file(uploaded_doc)
        if extracted:
            st.session_state.doc_content = extracted
            st.success(f"Loaded: {uploaded_doc.name}")

    st.divider()
    
    # Module E: Document Workspace & Download
    st.subheader("📝 Document Workspace")
    if st.session_state.created_doc:
        st.text_area("Generated Content:", value=st.session_state.created_doc, height=150)
        st.download_button(
            label="⬇️ Download Document (.txt)",
            data=st.session_state.created_doc,
            file_name="assistant_document.txt",
            mime="text/plain"
        )

# ----------------- MAIN CHAT UI -----------------
st.title("🤖 Multimodal Personal AI Assistant")
st.caption(f"Active Mode: **{mode}** | Powered by Gemini Flash")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
user_input = st.chat_input("Ask a question, analyze an attachment, or generate a document...")

if user_input:
    # Append and show user query
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build prompt based on selected mode and context
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                system_instruction = ""
                content_payload = []

                if mode == "Empathetic Emotional Support":
                    system_instruction = (
                        "You are an empathetic, non-judgmental personal emotional assistant. "
                        "Practice active listening, offer gentle reflections and coping strategies. "
                        "Do not diagnose conditions or claim to be a licensed therapist. "
                        "If self-harm or immediate crisis is mentioned, provide emergency helpline resources."
                    )
                elif mode == "Document RAG Intelligence" and st.session_state.doc_content:
                    system_instruction = (
                        "You are a document intelligence assistant. Answer the user question strictly using "
                        "the context extracted from the uploaded document below. If the answer cannot be found, "
                        "clearly state that the document does not contain that information.\n\n"
                        f"--- DOCUMENT CONTEXT ---\n{st.session_state.doc_content[:6000]}\n--- END CONTEXT ---"
                    )

                # Multimodal Image Input handling
                if uploaded_image:
                    img = Image.open(uploaded_image)
                    content_payload = [system_instruction, img, user_input]
                else:
                    full_prompt = f"{system_instruction}\n\nUser: {user_input}" if system_instruction else user_input
                    content_payload = [full_prompt]

                # Document creation command detection
                if any(k in user_input.lower() for k in ["create document", "generate report", "write a draft"]):
                    content_payload[0] += "\n\nPlease write this as a structured formal document suitable for export."

                response = model.generate_content(content_payload)
                reply_text = response.text

                # If document generation was requested, populate workspace
                if any(k in user_input.lower() for k in ["create document", "generate report", "write a draft"]):
                    st.session_state.created_doc = reply_text

                st.markdown(reply_text)
                st.session_state.messages.append({"role": "assistant", "content": reply_text})

            except Exception as e:
                st.error(f"Error: {e}")

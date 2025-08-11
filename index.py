import os
from dotenv import load_dotenv
import uuid
import base64
import io
import re
import pickle
import concurrent.futures

load_dotenv(dotenv_path=".env", override=True)
api_key = os.getenv("OPENAI_API_KEY")

import streamlit as st
import streamlit.components.v1 as components

## Providers
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
vector_store = InMemoryVectorStore(embeddings)

## OCR
import fitz
from PIL import Image

#######################################################################
# Remove footers and headers
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)
#######################################################################
## Utils
buffer_folder = "buffer"
buffer_name = "docs.pkl"
buffer_docs = f"{buffer_folder}/{buffer_name}"
UPLOAD_DIR = "data"

########################################
## Utils
def stt_util(audio):

    transcript = ''

    if audio:
        transcript = client.audio.transcriptions.create(
            model = "whisper-1",
            file = audio
        )

    return transcript.text

def llm_completion(input_text):

    output_text = ''

    response = client.chat.completions.create(
                model = "gpt-4o-mini",
                messages = [{"role": "user", "content": input_text}],
                temperature = 0,
            )

    output_text = response.choices[0].message.content

    return output_text

def tts_util(input_text):
    speech_file_path = "answer.mp3"

    # Check if the file exists, then remove it
    if os.path.exists(speech_file_path):
        os.remove(speech_file_path)

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=input_text
        ) as response:
    
        response.stream_to_file(speech_file_path)

    return speech_file_path

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        
        audio_html = f"""
                <audio id="player" controls autoplay>
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                <script>
                    var audio = document.getElementById("player");
                    audio.play();
                </script>
                """
        components.html(audio_html, height=100)

def pdf_to_base64_images(pdf_path: str) -> list[str]:
    pdf_document = fitz.open(pdf_path)
    base64_images = []

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        base64_images.append(base64_image)

    return base64_images

def display_base64_image(base64_str, caption=""):
    image_html = f'<img src="data:image/png;base64,{base64_str}" width="800"/>'
    st.markdown(image_html, unsafe_allow_html=True)
    if caption:
        st.caption(caption)

def clean_markdown_fences(text: str) -> str:
    # Remove triple backticks with optional language specifier
    cleaned = re.sub(r"```(?:\w+)?\n?", "", text)
    return cleaned.strip()

def base64_image_to_markdown(base64_str):

    query = """Extract all the text in the image as a markdown, including tables, headers and plain text.
    If you see any author or writer names, include a header saying "Authors" with the actual author information in it. If authors are not in text, don't include any header saying "Authors".
    If you find and image such as a diagram or other sort, create a description of the image.
    Do not use the word 'Markdown' or wrap the output in triple backticks. Avoid any code or markup formatting.
    markdown:
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"},
            },
        ],
    )
    response_temp = llm.invoke([message])
    response = clean_markdown_fences(response_temp.content)

    return response

def save_to_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_from_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)
    
def call_index():

    ## remove existing docs for indexing
    file_path = os.path.join(buffer_folder, buffer_name)
    try:
        os.remove(file_path)
    except Exception as e:
        pass

    ################################################
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    docs = []
    for pdf_path in pdf_files:
        ################################################
        # pdf_path = 'data/A human-centric drift controller framework.pdf'

        ## Converting pdf to images
        with st.spinner(f"{pdf_path} - Converting pdf to images..."):
            base64_images = pdf_to_base64_images(f"{UPLOAD_DIR}/{pdf_path}")

        ################################################
        ## Series
        # for i, img in enumerate(base64_images):
        #     ## Translating to text
        #     with st.spinner(f"{pdf_path} - Converting image {i+1} into text..."):
        #         temp_text = base64_image_to_markdown(img)
            
        #     ## appending document
        #     docs.append(Document(
        #         page_content=temp_text,
        #         metadata={"source": f"Page {i+1}"}
        #         ))

        ## Parallel
        def process_image(i_img_tuple):
            i, img = i_img_tuple
            temp_text = base64_image_to_markdown(img)
            return Document(
                page_content=temp_text,
                metadata={"source": f"Page {i+1}"}
            )

        try:
            # Use ThreadPoolExecutor to parallelize
            with st.spinner(f"{pdf_path} - Converting {len(base64_images)} images to text..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(executor.map(process_image, enumerate(base64_images)))

            st.info("Successful translation and store.")
        except Exception as e:
            st.error(f"Something went wrong while trying to convert from PDF to Markdown. {e}")

        # Collect documents
        docs.extend(list(results))

    ################################################
    ## Save data temporarily - Might be replaced by DB
    save_to_pickle(docs, buffer_docs)

def main():

    # Directory to store uploaded PDFs
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    st.title("📄 PDF Upload for RAG - Tutai")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Show file name and size
        st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")

        # Show save button
        if st.button("Save PDF"):
            # Define where to save the file
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            # Save the file
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.info(f"File saved to: `{save_path}`")

    ################################################
    # Make sure the folder exists
    if not os.path.exists(UPLOAD_DIR):
        st.warning(f"The folder `{UPLOAD_DIR}` does not exist.")
    else:
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]

        if pdf_files:
            st.write(f"Found {len(pdf_files)} PDF file(s):")
            for pdf in sorted(pdf_files):
                st.markdown(f"- 📄 `{pdf}`")
        else:
            st.info("No PDF files found in the folder.")

    if st.button("Index file(s)"):
        call_index()

if __name__ == '__main__':
    main()
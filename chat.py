import os
from dotenv import load_dotenv
import uuid
import base64
import io
import re
import pickle
import time
import json

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
from typing_extensions import List, TypedDict, Tuple
from langgraph.graph import START, StateGraph

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=TOP_K)
    # retrieved_docs = vector_store.similarity_search_with_relevance_scores(state["question"], k=TOP_K)
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
TOP_K = 4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

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
    If you see any author or writer names, include a header saying "Authors"
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

# Streamed response emulator
def response_generator():
    for word in st.session_state.temp_answer.split():
        yield word + " "
        time.sleep(0.05)

def main():

    buffer_docs = "buffer/docs.pkl"

    if 'temp_answer' not in st.session_state.keys():
        st.session_state.temp_answer = ''

    with st.spinner("Loading data and indexing..."):
        ################################################
        ## Chunking
        docs = load_from_pickle(buffer_docs)
        
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,  # chunk size (characters)
            chunk_overlap=CHUNK_OVERLAP,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)

        st.info(f"Split documents into {len(all_splits)} sub-documents.")
        # st.write(str(all_splits))

        ## Indexing
        document_ids = vector_store.add_documents(documents=all_splits)
        print(document_ids[:3])

        # ################################################
        ## Retrieval and Generation
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

    ## Init chat
    st.title("Tutai Bot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hi there! How can I help you today?"})

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    ####################################################
    # React to user input
    if prompt := st.chat_input("How can I help?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner(""):
                response = graph.invoke({"question": prompt})
            
            st.session_state.temp_answer = str(response['answer'])
            resp = st.write_stream(response_generator())

            with st.expander("Sources"):
                for d in response['context']:
                    st.write(f"**{d.metadata['source']}**")
                    st.markdown(d.page_content)
                    st.divider()
                    
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.temp_answer})

    # prompt = "What are the names of the authors?"
    # st.write(f"Question: {prompt}\n\n")

    # with st.spinner("Retrieving and Generating answer..."):
    #     result = graph.invoke({"question": prompt})
        
    #     st.write(f"Answer: {result['answer']}")
    #     st.write(f"Context: {result['context']}\n\n")


        
if __name__ == '__main__':
    main()
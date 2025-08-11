import os
from dotenv import load_dotenv
import uuid
import base64
import io
import re
import pickle
import time
import json
import pandas as pd

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

## Metrics
import evaluate as ev
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

# Load metrics
rouge = ev.load("rouge")
meteor = ev.load("meteor")


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

prompt.messages[0].prompt.template = """
You are an expert in aircraft engine maintenance, repair, and overhaul (MRO) processes, machine learning, and predictive analytics. You provide accurate, technical responses based on research findings.
Question: {question} 
Context: {context} 
Answer:
"""

# st.write(prompt.messages[0].prompt.template)

llm_model = "gpt-4.1-mini"
# llm_model = "ft:gpt-4.1-mini-2025-04-14:personal:clustering-paper:Byhehg4s"
llm_embeddings = "text-embedding-3-large"
TOP_K = 5
CHUNK_SIZE=7000
CHUNK_OVERLAP=4500

llm = ChatOpenAI(model=llm_model, api_key=api_key)
embeddings = OpenAIEmbeddings(model=llm_embeddings, api_key=api_key)
vector_store = InMemoryVectorStore(embeddings)
client = OpenAI(api_key=api_key)

## OCR
import fitz
from PIL import Image

buffer_docs = "buffer/docs.pkl"

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

    with st.expander("RAG Parameters:"):
        st.write(f"**LLM**: {llm_model}")
        st.write(f"**Embeddings Model**: {llm_embeddings}")
        st.write(f"**Top K Search**: {TOP_K}")
        st.write(f"**Chunk Size**: {CHUNK_SIZE}")
        st.write(f"**Overlap**: {CHUNK_OVERLAP}")

    ## Streamlit Session state
    if 'temp_answer' not in st.session_state.keys():
        st.session_state.temp_answer = ''
    if "indexing" not in st.session_state:
        st.session_state.indexing = False
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "gt_data" not in st.session_state:
        st.session_state.gt_data = None
    if "all_responses" not in st.session_state:
        st.session_state.all_responses = None
    
    index_button = st.button("Index and Vector Store document")
    if not st.session_state.indexing and index_button:
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
            st.session_state.graph = graph_builder.compile()

            st.session_state.indexing = True

    ## Init chat
    st.title("Tutai Evaluation Bot!")

    # st.write("All metrics available")
    # st.write(evaluate.list_evaluation_modules())

    # File uploader
    # File uploader
    uploaded_file = st.file_uploader("Upload a Test Q&A JSON file", type=["json"])

    if uploaded_file is not None:
        try:
            # Load the JSON file
            st.session_state.gt_data = json.load(uploaded_file)

            # Convert to DataFrame for better display
            df = pd.DataFrame(st.session_state.gt_data)
            questions = [item["question"] for item in st.session_state.gt_data]
            references = [item["answer"] for item in st.session_state.gt_data]

            # Show the table
            st.subheader("Questions and Answers")
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading JSON: {e}")
    else:
        st.info("Please upload a JSON file containing questions and answers.")

    ## Generate answers
    if st.button("Generate responses"):
        try:
            st.session_state.all_responses = []

            for prompt in questions:
                with st.spinner(f"Generating answer for: {prompt[:50]}..."):
                    response = st.session_state.graph.invoke({"question": prompt})
                    answer = response.get("answer", "") if isinstance(response, dict) else str(response)
                    st.session_state.all_responses.append(answer)

        except Exception as e:
            st.markdown("""
                        Ups, something went wrong. 
                     
                     Perhaps you forgot to:
                     1. Click 'Index and Vector Store document', or; 
                     2. Upload the tests document.""")

    ## Display generated responses
    if st.session_state.all_responses is not None:
        if len(st.session_state.all_responses) > 0:
            # Build DataFrame
            df = pd.DataFrame({
                "Question": questions,
                "Ground Truth": references,
                "LLM Answer": st.session_state.all_responses
            })

            st.subheader("Generated Responses")
            st.dataframe(df, use_container_width=True)

    st.divider()
    st.header("Evaluation options")
    col1, col2 = st.columns(2)
    with col1:
        button_stats = st.button("Evaluate with Statistics")
    with col2:
        button_llm = st.button("Evaluate with LLM")
    
    ## Evaluation
    if button_stats:
        try:
            with st.spinner("Evaluating with Statistics..."):
                # Compute metrics
                rouge_result = [
                    rouge.compute(predictions=[pred], references=[ref])
                    for pred, ref in zip(st.session_state.all_responses, references)
                ]
                rouge_1_scores = [r["rouge1"] for r in rouge_result]
                rouge_2_scores = [r["rouge2"] for r in rouge_result]

                meteor_scores = [
                    meteor.compute(predictions=[pred], references=[ref])["meteor"]
                    for pred, ref in zip(st.session_state.all_responses, references)
                ]

            # Build DataFrame
            df = pd.DataFrame({
                "Question": questions,
                "Ground Truth": references,
                "LLM Answer": st.session_state.all_responses,
                "ROUGE-1": rouge_1_scores,
                "ROUGE-2": rouge_2_scores,
                "METEOR": meteor_scores
            })

            st.subheader("Evaluation Results")
            st.dataframe(df, use_container_width=True)

            st.markdown("### 🔍 Evaluation Summary")
            st.write(f"**Average ROUGE-1**: {sum(rouge_1_scores)/len(rouge_1_scores):.4f}")
            st.write(f"**Average ROUGE-2**: {sum(rouge_2_scores)/len(rouge_2_scores):.4f}")
            st.write(f"**Average METEOR**: {sum(meteor_scores)/len(meteor_scores):.4f}")
        except Exception as e:
            st.markdown("""
                        Ups, something went wrong. 
                     
                     Perhaps you forgot to:
                     1. Click 'Index and Vector Store document', or; 
                     2. Upload the tests document, or; 
                     3. Generate the RAG system answers by clicking 'Generate responses'.""")

    if button_llm:
        correctness_metric_temp = GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            # NOTE: you can only provide either criteria or evaluation_steps, and not both
            # evaluation_steps=[
            #     "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            #     "You should also heavily penalize omission of detail",
            #     "Vague language, or contradicting OPINIONS, are OK"
            # ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        )

        results = []
        try:
            with st.spinner("LLM Evaluation...."):
                
                ## all
                all_test_cases = [LLMTestCase(input=ques, actual_output=pred, expected_output=ref) for pred, ref, ques in zip(st.session_state.all_responses, references, questions)]
                all_results = evaluate(test_cases=all_test_cases, metrics=[correctness_metric_temp])
                # st.write(all_results)

                all_scores = [a.metrics_data[0].score for a in all_results.test_results]

                df = pd.DataFrame({
                    "Question": [a.input for a in all_results.test_results],
                    "Ground Truth": [a.expected_output for a in all_results.test_results],
                    "LLM Answer": [a.actual_output for a in all_results.test_results],
                    "Success": [a.metrics_data[0].success for a in all_results.test_results],
                    "Score": all_scores,
                    "Reason": [a.metrics_data[0].reason for a in all_results.test_results]
                })

                st.subheader("LLM Evaluation")
                st.dataframe(df, use_container_width=True)

                st.markdown("### 🔍 Evaluation Summary")
                st.write(f"**Average Score**: {sum(all_scores)/len(all_scores):.4f}")
                
                # ## each
                # for pred, ref, ques in zip(st.session_state.all_responses, references, questions):
                #     test_case = LLMTestCase(
                #         input=ques,
                #         actual_output=pred,
                #         expected_output=ref
                #     )

                #     temp_eval = evaluate(test_cases=[test_case], metrics=[correctness_metric_temp])
                #     st.write(temp_eval.test_results[0].metrics_data[0].success)

                #     results.append(temp_eval)

                # st.write(results)
        except Exception as e:
            st.markdown("""
                        Ups, something went wrong. 
                     
                     Perhaps you forgot to:
                     1. Click 'Index and Vector Store document', or; 
                     2. Upload the tests document, or; 
                     3. Generate the RAG system answers by clicking 'Generate responses'.""")

if __name__ == '__main__':
    main()
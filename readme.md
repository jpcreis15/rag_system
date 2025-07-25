# Retrieval Augemnted Generation Example

This repository is for educational purposes so that people that fresh started into the LLM world, specifically learning and implementing RAG systems, can have a smooth understanding of it's inner workings through a small (and limited) prototype.

In this repo you find 3 dedicated Streamlit apps (this was done on purpose) for processing and indexing documents, chat with them through a RAG-based system and evaluating such system with statisical and llm-based metrics.

PDF to text:
- OCR done using LLM (gpt-4o-mini), from pdf image to markdown

Vector dataset:
- InMemoryVectorStore from LangChain

Evaluators:
- Evaluate from HuggingFace
- Deepeval from Confident AI

## Conda Env Configuration
If you're using conda for env management, here's how to configure:

Create env
```
conda create -n rag_system python=3.10
```
If you want, you can choose a different python version, e.g. 3.8 or 3.12

Activate env
```
conda activate rag_system
```

## Install dependencies
```
make install
```

## How it works
You have 3 python files with different purposes:

### index.py: 

Goal
- Upload a PDF document, store it and translate into text, and storing it into a python format to be further indexed using LangChain. Here, you can also see all documents in the folder that are going to be used for indexing. If there's one document, only this one is going to be translated to markdown and documents. If there's two, the two are going to be used.

Process:
1. You can first upload the document
2. Only when you click 'Save PDF' the document is stored in 'data' folder.
3. Only when you click 'Index file(s)' it is going to start the translation process to Markdown, and use LangChain's 'Document' to wrap all the content into documents. The python variable containing all the PDFs markdown Documents is persisted using pickle with the name 'docs.pkl' in folder 'buffer' for later use. This was done to avoid translating everytime you spin the Chat and Evaluate apps.
4. That's all for now. You should see a message 'Successful translation and store'. You can stop this app and spin one of the following two about Chat and Evaluate.

How to spin:
```
make run_index
```

### chat.py

Goal:
- Get all the text from the previous step (index.py) and perform chunking and overlaping on the data to be suitable for RAG. After this is done, you can now chat with the document. You can also enable the 'voice' which is quite nice! Try it!

Process:
1. First click 'Index and Vector Store document' to run the chunking and overlaping, and store into a vector database, ready to be used. The chunking size and overlap parameters used are the ones shown in the expander 'RAG Parameters'. Here you can see that the llm provider and embedding model are also defined, together with the 'top k search' that will search for the 'k' number of chunks in the vector database. To change such parameters, you should change the code accordingly and refresh the app in browser. After the indexing is done, you should see a meesage 'Split documents into X sub-documents'. This is the result of the chunking process and directly related to the defined chunking size. 

Parameters in code to be adapted according to your case:

```python
llm_model = "gpt-4o-mini"
llm_embeddings = "text-embedding-3-large"
TOP_K = 5
CHUNK_SIZE=5000
CHUNK_OVERLAP=2000
```

2. After you click 'Index and Vector Store document' the app is ready to be asked for your specific and challenging questions. Put the system to the test!
3. You can click on 'Voice enabled' to replace manual typing by voice interaction

How to spin:
```
make run_chat
```

### rag_evaluate.py

Goal:
- This in an extension of chat.py but instead of asking questions manually, we're asking automatically through a small dataset and evaluate the performance. The performance metris used are ROUGE-1, ROUGE-2, METEOR (statistical-based) and GEval (llm-based). The central idea is to evaluate how the RAG system implemented so far performs against a curated dataset.

Process:
1. Click 'Index and Vector Store document' to make the system 'RAG ready' for questioning as in the Chat example.
2. Then, the next step is to upload the dataset in json format with all the questions and ground truth answers for assessment. You have a sample dataset in folder 'tests'. Please mind that this json test is specific for a document, so it will miserably fail if you try to evaluate your specific documents. Please adapt the dataset to your needs. After you upload the json document, a small table is going to be shown with all the questions and answers.
3. Before you start the evaluation you should generate all the answer from the RAG system by clicking in 'Generate responses' button.
4. After this, you can choose to run the staticial-based or llm-based evaluations.
    1. If you click 'Evaluate with Statistics' a table is going to be shown with results per dataset entry and a final summary of the evaluation;
    2. If you click 'Evaluate with LLM' a table is going to be shown with results per dataset entry and a final summary of the evaluation. Here you can check the progress in the Terminal with a nice user interface;

How to spin:
```
make run_eval
```

## Additional documents

- requirements.txt: File with all the project dependencies
- Makefile: File to automate some of installation and execution steps
- .env_template: Since some of these implementations require a OpenAI API Key, this should be private to your project. Hence, you should create a file named '.env' with the same content as '.env_template' but with your corresponding key

**PS: Feel free to delete the files inside 'data' and 'buffer' folders. If you do that, please refresh your app in the browser.**
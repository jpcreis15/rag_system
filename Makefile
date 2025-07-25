install:
	python -m pip install -r requirements.txt

run_index:
	streamlit run index.py

run_chat:
	streamlit run chat.py

run_eval:
	streamlit run rag_evaluate.py
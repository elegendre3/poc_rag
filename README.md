# poc_rag
POC rag system

Inspiration: 
- https://github.com/svpino/llm
- https://www.youtube.com/watch?v=HRvyei7vFSM

# Env
Using poetry
poetry init --no-interaction
poetry run pip install -r requirements.txt 

# Notes
- Allow local opensource + API call (openai)



# Build

## To build and run the Bank Statement Analyzer:
- docker build -f Dockerfile_bank -t bank_statement_analyzer:0.0.1 .
- docker run -p 8501:8501 bank_statement_analyzer:0.0.1
- docker tag bank_statement_analyzer:0.0.1 elegendre3/bank_statement_analyzer:0.0.1
- docker push elegendre3/bank_statement_analyzer:0.0.1


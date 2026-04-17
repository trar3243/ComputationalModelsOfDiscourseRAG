# Computational Models of Discourse - RAG

## Initial considerations

* Set up your coding environment (if it isn't already)
* Install your IDE of choice — I like [VS Code](https://code.visualstudio.com/)
* Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (or equivalent Python package manager)
* Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
 * Configure [Git](https://git-scm.com/book/ms/v2/Getting-Started-First-Time-Git-Setup)
* Set up the environment locally: clone the [RAG-repo](https://github.com/trar3243/ComputationalModelsOfDiscourseRAG.git).

## Setup Instructions

### Environment
```
conda create -n discourse_project
source activate discourse_project
```
To activate and exiting the environment
```
conda activate discourse_project
conda deactivate
```

Install REQUIREMENTS (you only need to do this once)
```
pip install -r requirements.txt
```

## Getting an API Key

For this framework, we used a Gemini model (3.1-Flash) that can be accessed for free and allows 500 requests per day. 

Gemini API key can be obtained in [Google AI Studio](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://aistudio.google.com/app/apikey).

Once you have this, fill the .env file with your API key in quotes.

**IMPORTANT: DO NOT SHARE YOUR API KEY OR PUSH IT TO THE GITHUB**

## Running the scripts

In the first script (index_document.py), we do 4 steps in the RAG pipeline:
1. Loading .txt files from a directory
2. Chunking with LangChain
3. Getting local embeddings for the chunks
4. Storing embeddings in a Crhoma persistent collection

Usage:
- Store all .txt files in the 'documents' directory (create it if it doesn't exist).
- Run
```
python index_documents.py
```

In the second script (query_documents.py), we do other two steps in the RAG pipeline:

5. Retrieving relevant chunks from ChromaDB
6. Generating an answer using Gemini 3.1 Flash-Lite (it allows for 500 queries daily as of rn)

Usage:
```
python query_documents.py
```






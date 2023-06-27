# Generative Q&A
Using [Pinecone](https://docs.pinecone.io/docs/overview), [LangChain](https://langchain-langchain.vercel.app/) + [OpenAI](https://platform.openai.com/overview) for Generative Q&A with [Retrieval Augmented Generation (RAG)](https://ai.facebook.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/).


## Overview

1. Setup the knowledge base (in [Pinecone](https://www.pinecone.io/))
- Chunk the content
- Create vector embeddings from the chunks
- Load embeddings into a Pinecone index

2. Ask a question
- Create vector embedding of the question
- Find relevant context in Pinecone, looking for embeddings similar to the question
- Ask a question of OpenAI, using the relevant context from Pinecone

[TODO - add diagram]


## Setup

### Install dependencies
```console
pip install -r ./setup/requirements.txt
```

### Provide Pinecone & OpenAI API Keys
```console
cp dotenv .env
vi .env
```

### Use the notebooks to load the data into the Pinecone index (and run samnple queries)


[TODO - non-splade example is working, update the splade example]

[TODO - show sample output]


## Q&A App (using Streamlit)

### Install Dependencies
[TODO - clean up requirements.txt or pipenv]

### Run
```console
streamlit run streamlit-app.py
```
[TODO - show example screenshot]


## Next Steps

[TODO - Update to support multiple PDFs]

[TODO - add customization of look & feel]


## Background

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., … Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, & H. Lin (Eds.), **Advances in Neural Information Processing Systems** (Vol. 33, pp. 9459–9474). Retrieved from https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf

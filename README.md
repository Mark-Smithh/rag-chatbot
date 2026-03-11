# RAG Chatbot

Use LlamaIndex, Faiss, FaissVectorStore, and LLamaIndex ChatEngine to have a conversation where AI can answer questions with past context in mind.

## URLLIB Version

Using v1.25.1 because the following warning was seen when using the latest version 

NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020

```SHELL
pip3 install urllib3==1.25.1
```

## LlamaIndex Document Import

The tutorial import no longer works.  The corrected import is below

```SHELL
from llama_index.core import Document
```

## LlamaIndex

```SHELL
pip3 install llama_index
```

LlamaIndex is a data framework for LLM applications to ingest, structure, and access private or domain-specific data.

LlamaIndex provides a number of nice tools when it comes to working with LLMs on your data.
- Data connectors (for connecting to your various data sources)
- Indexing capabilities (we’ll see one of those a bit later)
- and more!

## Faiss (Facebook AI Simularity Search)

Faiss (Facebook AI Similarity Search) is a library that allows developers to quickly search for embeddings of multimedia documents that are similar to each other. It solves limitations of traditional query search engines that are optimized for hash-based searches, and provides more scalable similarity search functions.

Faiss itself is not a full vector database; it is a low‑level library for similarity search and clustering over dense vectors.

The recommended way to install Faiss is via Conda

```SHELL
brew install miniconda
conda install -c pytorch -c conda-forge faiss-cpu=1.14.1

### To install Faiss I had to do the following because the below error was returned when trying to install it
# Could not solve for environment specs The following packages are incompatible 
# ├─ faiss-cpu =1.14.1 * is installable and it requires │ └─ python >=3.12,<3.13.0a0 *, which can be installed; └─ pin on python =3.13 * is not installable because it requires └─ python =3.13 *, which conflicts with any installable versions previously reported. 
# Pins seem to be involved in the conflict. Currently pinned specs: - python=3.13
conda create -n faiss-env python=3.12
conda init zsh
conda activate faiss-env
conda install -c pytorch -c nvidia -c conda-forge faiss-cpu=1.14.1
```

## VectorStoreIndex

Vector Stores are a key component of retrieval-augmented generation (RAG) and so you will end up using them in nearly every application you make using LlamaIndex, either directly or indirectly.
Vector stores accept a list of Node objects and build an index from them.

## Embeddings

[Vector Embeddings](https://developers.openai.com/api/docs/guides/embeddings)

Embeddings are, in simplest terms, an easy way to convert words into numbers. The numbers that we convert text into represent a vector. The distiance between two vectors represents their similarity (small distance) or difference (large distance).

An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. Small distances suggest high relatedness and large distances suggest low relatedness.

## ServiceContext

llamaindex ServiceContext has been deprecated.  The new object is [Settings](https://developers.llamaindex.ai/python/framework/module_guides/supporting_modules/service_context_migration/).

## Running

If faiss library cannot load that means conda environment needs to be activated.

```SHELL
# the below is needed because 3.13 was installed and added to the path but faiss does not support python 3.13
# remove Python 3.13 from the path.  This makes global Python version 3.12.x.
# miniconda entries from path are below
# --- /opt/homebrew/Caskroom/miniconda/base/bin
# --- /opt/homebrew/Caskroom/miniconda/base/condabin

# remove miniconda entries from path 
export PATH=/opt/homebrew/opt/python@3.12/libexec/bin:/Users/marksmith/.local/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin:/opt/pmk/env/global/bin:/usr/local/MacGPG2/bin:/Applications/iTerm.app/Contents/Resources/utilities

cd Documents/tutorials/rag_chatbot
source .devenv2/bin/activate
conda activate faiss-env
python3 main.py

# (faiss-env) ((.devenv2) ) marksmith_18:33 rag_chatbot :-)
# python3 -c "import faiss; print(faiss.__version__)"
# conda deactivate
# conda activate faiss-env
```

## LlamaIndex ChatEngine

[ChatEngine](https://developers.llamaindex.ai/python/framework/module_guides/deploying/chat_engines/)

Chat engine is a high-level interface for having a conversation with your data (multiple back-and-forth instead of a single question & answer). Think ChatGPT, but augmented with your knowledge base.

Conceptually, it is a stateful analogy of a Query Engine. *By keeping track of the conversation history, it can answer questions with past context in mind.*

## LlamaIndex QueryEngine

[QueryEngine](https://developers.llamaindex.ai/python/framework/module_guides/deploying/query_engine)

Does not keep context.  Used when one question without followup questions are required.


[VectorStoreIndex](https://developers.llamaindex.ai/python/framework/module_guides/indexing/vector_store_index/)

[feedparser](https://feedparser.readthedocs.io/en/latest/)

[RAG is a Hack Podcast](https://www.latent.space/p/llamaindex)

[FAISS](https://ai.meta.com/tools/faiss/)

[ChatEngine](https://developers.llamaindex.ai/python/framework/module_guides/deploying/chat_engines/)

[QueryEngine](https://developers.llamaindex.ai/python/framework/module_guides/deploying/query_engine)

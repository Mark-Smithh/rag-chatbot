import feedparser
from dotenv import load_dotenv
from unstructured.partition.html import partition_html
from llama_index.core import Document
import faiss
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
import os

STORAGE_DIR="./storage"
FAISS_PATH = f"{STORAGE_DIR}/faiss.index"

def main():
    # get the ATOM feed link of the podcast.
    # use the feedparser library to parse the atom XML to something usable & structured.
    # Every Latent Space podcast includes an HTML based summary and in that summary is the transcript.
    # To get at that summary, we’re going to have to parse that HTML. 
    podcast_atom_link = "https://api.substack.com/feed/podcast/1084089.rss" # latent space podcast

    parsed = feedparser.parse(podcast_atom_link)
    episode = [ep for ep in parsed.entries if ep['title'] == "RAG Is A Hack - with Jerry Liu from LlamaIndex"][0]
    episode_summary = episode['summary']
    
    # set env variables from .env file
    load_dotenv()

    parsed_summary = partition_html(text=''.join(episode_summary))
    start_of_transcript = [x.text for x in parsed_summary].index("Transcript") + 1

    # LlamaIndex provides a document interface that allows us to convert our text into a Document object.
    pod_cast_transcript = [Document(text=t.text) for t in parsed_summary[start_of_transcript:]]

    # Load from another folder
    phone_data = SimpleDirectoryReader("./phone_data").load_data()

    documents = pod_cast_transcript + phone_data

    index = load_or_build_index(documents)
    
    # no_context(index)

    with_context(index)

def load_or_build_index(documents):
    # TO SAVE MONEY IF THE DOCUMENT EMBEDDINGS HAVE ALREADY BEEN PERSISTED TO DISK LOAD THEM
    if os.path.exists(FAISS_PATH) and os.path.exists(STORAGE_DIR):
        print("🔄 Loading existing FAISS index from disk...")
        # 1. Load FAISS binary index
        faiss_index = faiss.read_index(FAISS_PATH)
        # 2. Recreate the vector store
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        # 3. Load LlamaIndex metadata
        storage_context = StorageContext.from_defaults(
            persist_dir=STORAGE_DIR,
            vector_store=vector_store,
        )
        # 4. Rebuild the index
        return load_index_from_storage(storage_context)
    else:
        print("⚙️ Building new FAISS index...")
        # Faiss code below ... --------- 
        d = 1536 # dimensions of text-ada-embedding-002, the embedding model that we're going to use
        faiss_index = faiss.IndexFlatL2(d)
        embed_model = OpenAIEmbedding()

        Settings.llm = OpenAI(model="gpt-4.1-mini", temperature=0.0)
        Settings.embed_model = embed_model
        
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        # Persist LlamaIndex metadata
        index.storage_context.persist(persist_dir="./storage")
        # Persist FAISS binary index separately
        faiss.write_index(faiss_index, "./storage/faiss.index")
        return index
    
def with_context(index):
    # ***** USE CHAT ENGINE (chat engine keeps context)
    
    # Chat engine is a high-level interface for having a conversation with your data 
    # (multiple back-and-forth instead of a single question & answer). Think ChatGPT, but augmented with your knowledge base.
    # Conceptually, it is a stateful analogy of a Query Engine. 
    # By keeping track of the conversation history, it can answer questions with past context in mind.
    query = "What does Jerry think about RAG?"
    print(query+'\n')
    chat_eng = index.as_chat_engine(similarity_top_k=10, chat_mode='context')
    response = chat_eng.chat(query)
    print(response.response)
    print("# ---------------------------------------------------------------------------------")
    # print(chat_eng.chat_history)
    query_2 = "Who are we talking about?"
    print(query_2+'\n')
    response_2 = chat_eng.chat(query_2)
    print(response_2.response)
    print("# ---------------------------------------------------------------------------------")
    query_3 = "What is Mark's mobile phone number?"
    print(query_3+'\n')
    response_3 = chat_eng.chat(query_3)
    print(response_3.response)
    print("# ---------------------------------------------------------------------------------")
    query_5 = "What is Mark's home phone number?"
    print(query_5+'\n')
    response_5 = chat_eng.chat(query_5)
    print(response_5.response)
    print("# ---------------------------------------------------------------------------------")
    query_7 = "What is Joe's mobile phone number?"
    print(query_7+'\n')
    response_7 = chat_eng.chat(query_7)
    print(response_7.response)
    print("# ---------------------------------------------------------------------------------")

def no_context(index):
    # ***** QUERY ENGINE (query engine does not keep context)
    query = "What does Jerry think about RAG?"
    print(query+'\n')
    response = index.as_query_engine(similarity_top_k=10).query(query)
    print(response.response)
    print("# ---------------------------------------------------------------------------------")
    query_2 = "Who are we talking about?"
    print(query_2+'\n')
    response_2 = index.as_query_engine(similarity_top_k=10).query(query_2)
    print(response_2.response)
    print("# ---------------------------------------------------------------------------------")
    query_3 = "What is Mark's mobile phone number?"
    print(query_3+'\n')
    response_3 = index.as_query_engine(similarity_top_k=1).query(query_3)
    print(response_3.response)
    print("# ---------------------------------------------------------------------------------")
    query_5 = "What is Mark's home phone number?"
    print(query_5+'\n')
    response_5 = index.as_query_engine(similarity_top_k=1).query(query_5)
    print(response_5.response)
    print("# ---------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
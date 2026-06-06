import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter as RecCharTxtSpltr
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, request, jsonify


def initialize_embeddings(llmod_api_key: str) -> OpenAIEmbeddings:
    embeddings = OpenAIEmbeddings(
        api_key=llmod_api_key,
        base_url="https://api.llmod.ai",
        model="4UHRUIN-text-embedding-3-small"
    )
    return embeddings


def retrieve_relevant_chunks(
    query: str, embeddings: OpenAIEmbeddings, pinecone_api_key: str, index_name: str, top_k: int = 3) -> list[tuple[Document, float]]: 
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return results


def generate_rag_response(
    query: str, retrieved_results: list[tuple[Document, float]], llmod_api_key: str) -> dict: 
    formatted_context = ""
    context_list = []  
    for i, (chunk, score) in enumerate(retrieved_results):
        title = chunk.metadata.get("title", "Unknown Title")
        authors = chunk.metadata.get("authors", "Unknown Authors")
        url = chunk.metadata.get("url", f"id_{i}") 
        text = chunk.page_content
        formatted_context += f"--- Article {i+1} ---\nTitle: {title}\nAuthor(s): {authors}\nText: {text}\n\n"
        context_list.append({
            "article_id": url, 
            "title": title,
            "chunk": text,
            "score": float(score) 
        })
    
    llm = ChatOpenAI(
        api_key=llmod_api_key,
        base_url="https://api.llmod.ai",
        model="4UHRUIN-gpt-5-mini"
    )
    system_prompt = (
        "You are a Medium-article assistant that answers questions strictly and only "
        "based on the Medium articles dataset context provided to you (metadata "
        "and article passages). You must not use any external knowledge, the open "
        "internet, or information that is not explicitly contained in the retrieved "
        "context. If the answer cannot be determined from the provided context, "
        "respond: \"I don't know based on the provided Medium articles data.\" "
        "Always explain your answer using the given context, quoting or "
        "paraphrasing the relevant article passage or metadata when helpful."
    )
    user_prompt_string = f"Context:\n{formatted_context}\n\nQuestion: {query}"
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{user_prompt}") 
    ])
    chain = prompt_template | llm
    response = chain.invoke({
        "user_prompt": user_prompt_string
    })
    final_output = {
        "response": response.content,
        "context": context_list,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt_string
        }
    }
    return final_output
    

if __name__ == '__main__':
    # Initialize the Flask web application
    app = Flask(__name__)
    TOP_K = 17  

    # ENDPOINT 1: /api/prompt - Receives the user question and returns the RAG response using a POST request
    @app.route('/api/prompt', methods=['POST'])
    def handle_prompt():
        incoming_data = request.get_json()
        user_question = incoming_data.get("question", "")

        llmod_key = os.environ.get("LLMOD_API_KEY") 
        pinecone_key = os.environ.get("PINECONE_API_KEY")
        pinecone_index_name = "individual-rag-task-index-final"
        
        embeddings_model = initialize_embeddings(llmod_api_key=llmod_key)
        retrieved_results = retrieve_relevant_chunks(
            query=user_question,
            embeddings=embeddings_model,
            pinecone_api_key=pinecone_key,
            index_name=pinecone_index_name,
            top_k=TOP_K
        )
        final_response_dict = generate_rag_response(
            query=user_question,
            retrieved_results=retrieved_results,
            llmod_api_key=llmod_key
        )
        return jsonify(final_response_dict)
    
    # ENDPOINT 2: /api/stats - Returns the hyperparameters configuration using a GET request
    @app.route('/api/stats', methods=['GET'])
    def get_stats():
        stats_dict = {
            "chunk_size": 700,
            "overlap_ratio": 0.285, 
            "top_k": TOP_K
        }
        return jsonify(stats_dict)
    
    app.run(port=5000, debug=True)




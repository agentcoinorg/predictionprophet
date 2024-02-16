from chromadb import Collection, EphemeralClient
import chromadb.utils.embedding_functions as embedding_functions

from evo_researcher.models.WebScrapeResult import WebScrapeResult

def create_embeddings_from_results(results: list[WebScrapeResult], text_splitter, api_key: str) -> Collection:
    client = EphemeralClient()
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-ada-002"
            )
    collection = client.create_collection(
        name="web_search_results",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    texts = []
    metadatas = []

    for scrape_result in results:
        text_splits = text_splitter.split_text(scrape_result.content)
        texts += text_splits
        metadatas += [scrape_result.dict() for _ in text_splits]
        
    print(f"Created {len(texts)} embeddings")
    print(f"Created {len(metadatas)} metadatas")      

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=[f'id{i}' for i in range(len(texts))]
    )
    return collection
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from evo_researcher.models.WebScrapeResult import WebScrapeResult


def create_embeddings_from_results(results: list[WebScrapeResult], text_splitter, api_key: str) -> Chroma:
    collection = Chroma(embedding_function=OpenAIEmbeddings(openai_api_key=api_key))
    texts = []
    metadatas = []

    for scrape_result in results:
        text_splits = text_splitter.split_text(scrape_result.content)
        texts += text_splits
        metadatas += [scrape_result.dict() for _ in text_splits]
        
    print(f"Created {len(texts)} embeddings")
    print(f"Created {len(metadatas)} metadatas")      

    collection.add_texts(
        texts=texts,
        metadatas=metadatas
    )
    return collection
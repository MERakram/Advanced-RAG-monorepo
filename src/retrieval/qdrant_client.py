import os
from typing import List, Dict, Any, Optional
from langfuse import observe  # SDK v3
from qdrant_client import QdrantClient, models
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class QdrantRetriever:
    """
    Manages Qdrant interactions: Collection setup, Ingestion, and Hybrid Retrieval.
    """
    
    def __init__(self, collection_name: str = "enterprise_rag"):
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        
        # Connection retry logic
        import time
        max_retries = 12
        retry_interval = 5
        for i in range(max_retries):
            try:
                # Simple check to see if qdrant is responsive
                self.client.get_collections()
                print(f"Successfully connected to Qdrant (attempt {i+1}/{max_retries})")
                break
            except Exception as e:
                print(f"Qdrant connection failed (attempt {i+1}/{max_retries}): {e}")
                if i < max_retries - 1:
                    time.sleep(retry_interval)
                else:
                    print("Could not connect to Qdrant after multiple attempts.")
        
        # Use local HuggingFace embedding model (no API key required)
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Creates the collection with Dense and Sparse vector configuration if it doesn't exist.
        """
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384,  # all-MiniLM-L6-v2 embedding dimension
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                }
            )

    def upsert_nodes(self, nodes: List[BaseNode]):
        """
        Generates embeddings (Dense + Sparse) and uploads nodes to Qdrant.
        """
        points = []
        for node in nodes:
            # Generate Dense Embedding
            dense_vector = self.embed_model.get_text_embedding(node.get_content())
            
            # Generate Sparse Vector (BM25-like)
            # Note: For true BM25 in Qdrant, you often compute it client-side or use a model like SPLADE.
            # For simplicity here, we'll use a placeholder or a simple frequency map if not using a specific model.
            # In a real production setup, use 'prithivida/Splade_PP_en_v1' or similar.
            sparse_vector = self._compute_sparse_vector(node.get_content())

            points.append(models.PointStruct(
                id=node.node_id,
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector,
                },
                payload=node.metadata or {}
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Upserted {len(points)} points to Qdrant.")

    def _compute_sparse_vector(self, text: str) -> models.SparseVector:
        """
        Simple frequency-based sparse vector for demonstration. 
        In production, use a proper SPLADE model.
        """
        # This is a naive implementation. 
        # TODO: Integrate a proper SPLADE model for high-quality sparse vectors.
        from collections import Counter
        tokens = text.lower().split()
        counts = Counter(tokens)
        
        # Map tokens to arbitrary indices (hashing) for demo purposes
        indices = [hash(token) % 100000 for token in counts.keys()]
        values = list(counts.values())
        
        return models.SparseVector(indices=indices, values=values)

    @observe(name="qdrant_search")
    def search(self, query: str, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search (Dense + Sparse).
        """
        # 1. Generate Query Embeddings
        query_dense = self.embed_model.get_text_embedding(query)
        query_sparse = self._compute_sparse_vector(query)

        # 2. Search
        # Qdrant supports hybrid search via prefetch or fusion. 
        # Here we simply search both and will merge/rerank later.
        
        # Dense Search
        dense_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_dense),
            limit=limit,
            with_payload=True
        )
        
        # Sparse Search
        sparse_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("sparse", query_sparse),
            limit=limit,
            with_payload=True
        )
        
        # Combine results (simple deduplication by ID)
        seen_ids = set()
        combined = []
        
        for res in dense_results + sparse_results:
            if res.id not in seen_ids:
                combined.append({
                    "id": res.id,
                    "score": res.score,
                    "payload": res.payload,
                    "text": res.payload.get("text", "") # Assuming text is stored in payload
                })
                seen_ids.add(res.id)
                
        return combined

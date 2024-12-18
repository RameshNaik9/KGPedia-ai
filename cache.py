from functools import lru_cache
from typing import List, Dict
from llama_index.core.schema import NodeWithScore
import time

class NodeCache:
    def __init__(self, max_size: int = 20, ttl: int = 3600):
        self.cache: Dict[str, tuple[List[NodeWithScore], float]] = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
    
    def get(self, query: str) -> List[NodeWithScore] | None:
        if query in self.cache:
            nodes, timestamp = self.cache[query]
            if time.time() - timestamp <= self.ttl:
                return nodes
            else:
                del self.cache[query]  # Remove expired entry
        return None
    
    def put(self, query: str, nodes: List[NodeWithScore]):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_query = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_query]
        self.cache[query] = (nodes, time.time())
from . import bm25, embedding_rag, long_context

METHODS = {"lc": long_context.build_context, "bm25": bm25.build_context, "emb": embedding_rag.build_context}

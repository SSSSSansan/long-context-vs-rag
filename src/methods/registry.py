from . import bm25, embedding_rag, long_context, no_context

METHODS = {"lc": long_context.build_context, "bm25": bm25.build_context, "emb": embedding_rag.build_context,
           "nocontext": no_context.build_context}

# методы без параметра k: LC кладёт весь корпус, NO-CONTEXT не кладёт ничего
NO_K_METHODS = ("lc", "nocontext")

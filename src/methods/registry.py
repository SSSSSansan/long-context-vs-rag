from . import bm25, embedding_rag, hybrid, long_context, no_context
from .agent import INDEXES as AGENT_INDEXES

METHODS = {"lc": long_context.build_context, "bm25": bm25.build_context, "emb": embedding_rag.build_context,
           "nocontext": no_context.build_context,
           "hyb_rrf": hybrid.build_rrf, "hyb_rerank": hybrid.build_rerank}

# методы без параметра k: LC кладёт весь корпус, NO-CONTEXT не кладёт ничего
NO_K_METHODS = ("lc", "nocontext", *AGENT_INDEXES)   # agent_*: k выбирает сама модель в search_corpus

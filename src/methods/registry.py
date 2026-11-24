from functools import partial

from . import bm25, embedding_rag, hybrid, long_context, no_context
from .agent import INDEXES as AGENT_INDEXES

METHODS = {"lc": long_context.build_context, "bm25": bm25.build_context, "emb": embedding_rag.build_context,
           "nocontext": no_context.build_context,
           "hyb_rrf": hybrid.build_rrf, "hyb_rerank": hybrid.build_rerank,
           # постфактум-проверки (REPORT.md, «Бонус-эксперименты»); в главную таблицу не входят
           "emb_bge": embedding_rag.build_context_bge,                          # эмбеддинги BAAI/bge-m3 вместо text-embedding-3-small
           "bm25_doc": partial(bm25.build_context, order="document"),           # OP-RAG: чанки в порядке документа
           "hyb_rerank_doc": hybrid.build_rerank_doc}

# методы без параметра k: LC кладёт весь корпус, NO-CONTEXT не кладёт ничего
NO_K_METHODS = ("lc", "nocontext", *AGENT_INDEXES)   # agent_*: k выбирает сама модель в search_corpus

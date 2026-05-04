"""
QA Chain
=========
Question-answering pipeline using Ollama LLM grounded by
knowledge graph context retrieved via the hybrid retriever.
"""

import logging
from typing import Optional
from pathlib import Path
import ollama
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OLLAMA_MODEL
from rag.retriever import HybridRetriever
from graph.graph_manager import KnowledgeGraph, get_graph

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = """You are MicroKG Assistant, an expert AI on microplastics research.
You answer questions using ONLY the provided knowledge graph context.

RULES:
1. Base your answer ONLY on the provided context. Do not use prior knowledge.
2. If the context doesn't contain enough information, say so clearly.
3. Cite specific entities and relationships from the context.
4. Be concise but thorough.
5. Structure your answer with clear sections if the question is complex.
"""


class QAChain:
    """Knowledge-grounded QA pipeline using Ollama + KG retrieval."""

    def __init__(self, kg: Optional[KnowledgeGraph] = None, model: str = ""):
        self.kg = kg or get_graph()
        self.model = model or OLLAMA_MODEL
        self.retriever = HybridRetriever(self.kg)

    def ask(self, question: str, top_k: int = 10, hops: int = 1) -> dict:
        """
        Answer a question using the knowledge graph.

        Args:
            question: Natural language question
            top_k: Number of retrieval results
            hops: Graph traversal depth

        Returns:
            Dict with answer, context, and sources
        """
        # Retrieve context
        retrieval = self.retriever.retrieve(question, top_k=top_k, hops=hops)
        context = retrieval["combined_context"]

        if not context.strip():
            return {
                "question": question,
                "answer": "I don't have enough information in the knowledge graph to answer this question.",
                "context": "",
                "sources": [],
            }

        # Build prompt
        user_prompt = f"""CONTEXT FROM KNOWLEDGE GRAPH:
{context}

QUESTION: {question}

Answer based ONLY on the above context. Cite specific entities and relationships."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.3, "num_predict": 1024},
            )
            answer = response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"QA generation failed: {e}")
            answer = f"Error generating answer: {e}"

        # Extract source entities
        sources = [
            {"name": r.get("name", ""), "type": r.get("entity_type", ""), "score": r.get("score", 0)}
            for r in retrieval.get("vector_matches", [])[:5]
        ]

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "sources": sources,
            "triples_used": len(retrieval.get("triples", [])),
        }

    def ask_interactive(self):
        """Interactive QA loop in the terminal."""
        print("\n" + "=" * 60)
        print("  🔬 MicroKG Question Answering System")
        print("  Type 'quit' to exit, 'stats' for graph stats")
        print("=" * 60)

        while True:
            question = input("\n❓ Your question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if question.lower() == "stats":
                stats = self.kg.get_stats()
                print(f"\n📊 Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
                for t, c in stats.get("nodes_by_type", {}).items():
                    print(f"   {t}: {c}")
                continue
            if not question:
                continue

            print("\n🔍 Searching knowledge graph...")
            result = self.ask(question)
            print(f"\n💡 Answer:\n{result['answer']}")
            if result["sources"]:
                print(f"\n📚 Sources: {', '.join(s['name'] for s in result['sources'][:3])}")
            print(f"   (Used {result['triples_used']} KG triples)")

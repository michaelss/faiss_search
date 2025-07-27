"""
Utility functions for the PDF search system.
"""

from typing import List, Tuple
from langchain.schema import Document

def format_qa_results(answer: str, source_docs: List[Document], question: str):
    """
    Formata e exibe os resultados da pergunta-resposta

    Args:
        answer: A resposta gerada pelo LLM
        source_docs: Os documentos de origem utilizados
        question: A pergunta original
    """
    print(f"\n{'='*80}")
    print(f"PERGUNTA: {question}")
    print(f"{'='*80}")

    print(f"\n🤖 RESPOSTA (Azure OpenAI):")
    print(f"{'─'*60}")
    print(answer)

    if source_docs:
        print(f"\n📚 FONTES CONSULTADAS:")
        print(f"{'─'*60}")

        for i, doc in enumerate(source_docs, 1):
            print(f"\n[FONTE {i}]")
            print(f"Arquivo: {doc.metadata.get('source_file', 'Desconhecido')}")
            print(f"Página: {doc.metadata.get('page', 'N/A')}")

            # Mostra um trecho do documento
            content = doc.page_content.strip()
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"Trecho: {content}")

    print(f"\n{'='*80}")

def format_results(results: List[Tuple[Document, float]], query: str):
    """
    Formata e exibe os resultados da busca

    Args:
        results: Os resultados da busca
        query: A consulta original
    """
    if not results:
        print("Nenhum resultado encontrado.")
        return

    print(f"\n{'='*80}")
    print(f"RESULTADOS PARA: '{query}'")
    print(f"{'='*80}")

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[RESULTADO {i}] - Similaridade: {1-score:.4f}")
        print(f"Arquivo: {doc.metadata.get('source_file', 'Desconhecido')}")
        print(f"Página: {doc.metadata.get('page', 'N/A')}")
        print(f"{'─'*60}")

        # Limita o texto exibido
        content = doc.page_content.strip()
        if len(content) > 500:
            content = content[:500] + "..."

        print(f"Conteúdo:\n{content}")
        print(f"{'─'*60}")


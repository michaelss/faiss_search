
import config

from dotenv import load_dotenv

from search_system import PDFSearchSystem
from utils import format_results, format_qa_results

def main():
    """
    Main function of the program.
    """
    load_dotenv()
    search_system = PDFSearchSystem(use_azure_llm=True)

    print("Busca Semântica em PDFs com Azure OpenAI")
    print("="*60)

    print("\nCarregando índice existente...")
    search_system.load_index()

    while True:
        print("\nOpções:")
        print("1. Criar novo índice")
        print("4. Realizar busca simples (similaridade)")
        print("5. Fazer pergunta (Azure OpenAI + RAG)")
        print("6. Sair")

        choice = input("\nEscolha uma opção (1-6): ").strip()

        if choice == "1":
            print("\nCriando novo índice...")
            sources = config.SOURCES

            if not sources or all(not s for s in sources):
                print("\nNenhuma fonte fornecida.")
                continue

            print("Fontes a serem indexadas:")
            for source in sources:
                print(f"  - {source}")

            success = search_system.create_index(sources)
            if success:
                print("Índice criado com sucesso!")
                print("\nCarregando índice existente...")
                search_system.load_index()

        elif choice == "4":
            if not search_system.vectorstore:
                print("\nÍndice não carregado. Por favor, carregue ou crie um índice primeiro.")
                continue

            print("\nModo de Busca Simples (Similaridade)")
            print("Digite 'sair' para retornar ao menu principal")

            while True:
                query = input("\nDigite sua consulta: ").strip()

                if query.lower() in ['sair', 'exit', 'quit']:
                    break

                if not query:
                    print("Por favor, digite uma consulta válida.")
                    continue

                print(f"\nBuscando por: '{query}'...")
                results = search_system.search(query)
                format_results(results, query)

        elif choice == "5":
            if not search_system.qa_chain:
                if search_system.vectorstore:
                    success = search_system.setup_qa_chain()
                    if success:
                        print("Azure OpenAI Q&A configurado!")
                    else:
                        print("Falha na configuração. Verifique suas variáveis de ambiente.")
                else:
                    print("Carregue um índice primeiro (opções 1 ou 2).")

            print("\nModo Pergunta-Resposta com Azure OpenAI")
            print("Digite 'sair' para retornar ao menu principal")

            while True:
                question = input("\nFaça sua pergunta: ").strip()

                if question.lower() in ['sair', 'exit', 'quit']:
                    break

                if not question:
                    print("Por favor, digite uma pergunta válida.")
                    continue

                answer, source_docs = search_system.ask_question(question)
                format_qa_results(answer, source_docs, question)

        elif choice == "6":
            print("Encerrando o programa...")
            break

        else:
            print("Opção inválida. Por favor, tente novamente.")

if __name__ == "__main__":
    main()

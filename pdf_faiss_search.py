#!/usr/bin/env python3
"""
Sistema de Busca Semântica em PDFs usando FAISS e LangChain
Autor: Assistant
Descrição: Indexa PDFs e permite busca semântica usando embeddings
"""

import os
import sys
import pickle
from typing import List, Dict, Tuple
from pathlib import Path

try:
    # import faiss
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.schema import Document
    from langchain.chains import RetrievalQA
    from langchain.llms import AzureOpenAI
except ImportError as e:
    print(f"Erro de importação: {e}")
    print("Execute: pip install langchain pypdf sentence-transformers")
    sys.exit(1)

class PDFSearchSystem:
    def __init__(self, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa o sistema de busca em PDFs
        
        Args:
            embeddings_model: Modelo de embeddings a ser usado
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.vectorstore = None
        self.documents = []
        self.index_path = "faiss_index"
        self.metadata_path = "documents_metadata.pkl"
        
    def load_and_split_pdfs(self, pdf_paths: List[str], chunk_size: int = 1000, 
                           chunk_overlap: int = 200) -> List[Document]:
        """
        Carrega e divide os PDFs em chunks
        
        Args:
            pdf_paths: Lista de caminhos para os arquivos PDF
            chunk_size: Tamanho dos chunks de texto
            chunk_overlap: Sobreposição entre chunks
            
        Returns:
            Lista de documentos processados
        """
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Aviso: Arquivo {pdf_path} não encontrado. Pulando...")
                continue
                
            try:
                print(f"Carregando PDF: {pdf_path}")
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                
                # Adiciona metadados do arquivo
                for page in pages:
                    page.metadata['source_file'] = os.path.basename(pdf_path)
                    page.metadata['full_path'] = pdf_path
                
                # Divide em chunks
                chunks = text_splitter.split_documents(pages)
                documents.extend(chunks)
                print(f"  Processado: {len(chunks)} chunks extraídos")
                
            except Exception as e:
                print(f"Erro ao processar {pdf_path}: {e}")
                continue
        
        return documents
    
    def create_index(self, pdf_paths: List[str]) -> bool:
        """
        Cria o índice FAISS a partir dos PDFs
        
        Args:
            pdf_paths: Lista de caminhos para os arquivos PDF
            
        Returns:
            True se o índice foi criado com sucesso
        """
        print("Iniciando indexação dos PDFs...")
        
        # Carrega e processa documentos
        self.documents = self.load_and_split_pdfs(pdf_paths)
        
        if not self.documents:
            print("Nenhum documento foi carregado. Verifique os caminhos dos PDFs.")
            return False
        
        print(f"Total de chunks processados: {len(self.documents)}")
        
        # Cria o índice FAISS
        print("Criando índice FAISS...")
        try:
            self.vectorstore = FAISS.from_documents(
                documents=self.documents,
                embedding=self.embeddings
            )
            
            # Salva o índice
            self.save_index()
            print(f"Índice criado e salvo com sucesso!")
            return True
            
        except Exception as e:
            print(f"Erro ao criar índice: {e}")
            return False
    
    def save_index(self):
        """Salva o índice FAISS e metadados"""
        if self.vectorstore:
            self.vectorstore.save_local(self.index_path)
            
            # Salva metadados dos documentos
            metadata = [doc.metadata for doc in self.documents]
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
    
    def load_index(self) -> bool:
        """
        Carrega um índice FAISS existente
        
        Returns:
            True se o índice foi carregado com sucesso
        """
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self.vectorstore = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Carrega metadados
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                print(f"Índice carregado com sucesso! {len(metadata)} documentos indexados.")
                return True
            else:
                print("Índice não encontrado. Execute a indexação primeiro.")
                return False
                
        except Exception as e:
            print(f"Erro ao carregar índice: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Realiza busca semântica no índice
        
        Args:
            query: Consulta do usuário
            k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (documento, score de similaridade)
        """
        if not self.vectorstore:
            print("Índice não carregado. Carregue ou crie um índice primeiro.")
            return []
        
        try:
            # Busca com scores de similaridade
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return results
            
        except Exception as e:
            print(f"Erro na busca: {e}")
            return []
    
    def format_results(self, results: List[Tuple[Document, float]], query: str):
        """
        Formata e exibe os resultados da busca
        
        Args:
            results: Resultados da busca
            query: Consulta original
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
    
    def setup_qa_chain(self, llm_model="gpt-3.5-turbo"):
        """Configura chain de pergunta-resposta"""
        if not self.vectorstore:
            print("Índice não carregado. Carregue ou crie um índice primeiro.")
            return

        llm = AzureOpenAI(model=llm_model, 
                          deployment_name="gpt-35-turbo",
                          temperature=0.2,
                          api_version="2023-05-15",
                          base_url="",
                          api_key="")
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    def ask_question(self, question: str):
        """Responde pergunta usando LLM + contexto dos PDFs"""
        result = self.qa_chain({"query": question})
        return result["result"], result["source_documents"]


def main():
    """Função principal do programa"""
    # Exemplos de PDFs fictícios (substitua pelos seus arquivos reais)
    pdf_files = [
        "pdfs/459.pdf",  # Ex: Manual de Python
        "pdfs/480.pdf",  # Ex: Guia de Machine Learning  
        "pdfs/494.pdf"   # Ex: Documentação de APIs
    ]
    
    # Inicializa o sistema
    search_system = PDFSearchSystem()
    
    print("Sistema de Busca Semântica em PDFs")
    print("="*50)
    
    # Menu principal
    while True:
        print("\nOpções:")
        print("1. Criar novo índice")
        print("2. Carregar índice existente")
        print("3. Realizar busca")
        print("4. Perguntar ao LLM (GPT-3.5) com contexto dos PDFs")
        print("5. Sair")
        
        choice = input("\nEscolha uma opção (1-4): ").strip()
        
        if choice == "1":
            print("\nCriando novo índice...")
            print("Arquivos a serem indexados:")
            for pdf in pdf_files:
                print(f"  - {pdf}")
            
            # Verifica se arquivos existem
            existing_files = [f for f in pdf_files if os.path.exists(f)]
            if not existing_files:
                print("\nNenhum arquivo PDF encontrado!")
                print("Por favor, coloque os seguintes arquivos no diretório atual:")
                for pdf in pdf_files:
                    print(f"  - {pdf}")
                continue
            
            success = search_system.create_index(pdf_files)
            if success:
                print("Índice criado com sucesso!")
            
        elif choice == "2":
            print("\nCarregando índice existente...")
            search_system.load_index()
            
        elif choice == "3":
            if not search_system.vectorstore:
                print("\nÍndice não carregado. Carregue ou crie um índice primeiro.")
                continue
            
            print("\nModo de Busca Ativado")
            print("Digite 'sair' para voltar ao menu principal")
            
            while True:
                query = input("\nDigite sua consulta: ").strip()
                
                if query.lower() in ['sair', 'exit', 'quit']:
                    break
                
                if not query:
                    print("Por favor, digite uma consulta válida.")
                    continue
                
                print(f"\nBuscando por: '{query}'...")
                results = search_system.search(query, k=3)
                search_system.format_results(results, query)
        

        elif choice == "4":
            if not search_system.vectorstore:
                print("\nÍndice não carregado. Carregue ou crie um índice primeiro.")
                continue
            
            search_system.setup_qa_chain()
            print("\nModo Pergunta-Resposta Ativado")
            print("Digite 'sair' para voltar ao menu principal")
            
            while True:
                question = input("\nDigite sua pergunta: ").strip()
                
                if question.lower() in ['sair', 'exit', 'quit']:
                    break
                
                if not question:
                    print("Por favor, digite uma pergunta válida.")
                    continue
                
                print(f"\nPerguntando: '{question}'...")
                answer, sources = search_system.ask_question(question)
                
                print(f"\nResposta: {answer}")
                if sources:
                    print("\nFontes utilizadas:")
                    for doc in sources:
                        print(f"  - {doc.metadata.get('source_file', 'Desconhecido')} (Página {doc.metadata.get('page', 'N/A')})")

        elif choice == "5":
            print("Encerrando programa...")
            break
        
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
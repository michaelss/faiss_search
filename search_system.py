"""
PDF Search System
"""

import os
import pickle
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup
import validators

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from core.config import settings

class PDFSearchSystem:
    """
    A system for searching and querying PDFs using a FAISS vector store and LangChain.
    """
    def __init__(self, use_azure_llm: bool = False):
        """
        Initializes the PDFSearchSystem.

        Args:
            use_azure_llm: Whether to use Azure OpenAI for Q&A.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL)
        self.vectorstore = None
        self.documents = []
        self.index_path = settings.INDEX_PATH
        self.metadata_path = settings.METADATA_PATH
        self.qa_chain = None
        self.use_azure_llm = use_azure_llm

        if self.use_azure_llm:
            self.setup_azure_llm()

    def load_and_split_files(self, file_paths: List[str]) -> List[Document]:
        """
        Loads and splits files into chunks.

        Args:
            file_paths: A list of paths to the files.

        Returns:
            A list of Document objects.
        """
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )

        for dir_path in file_paths:
            if not os.path.exists(dir_path):
                print(f"Aviso: Diretório não encontrado: {dir_path}. Pulando...")
                continue

            try:
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    if filename.lower().endswith(".pdf"):
                        print(f"Carregando PDF: {file_path}")
                        loader = PyPDFLoader(file_path)
                        pages = loader.load()
                    elif filename.lower().endswith(".txt"):
                        print(f"Carregando arquivo de texto: {file_path}")
                        loader = TextLoader(file_path)
                        pages = loader.load()
                    else:
                        continue

                    for page in pages:
                        page.metadata['source_file'] = os.path.basename(file_path)
                        page.metadata['full_path'] = file_path

                    chunks = text_splitter.split_documents(pages)
                    documents.extend(chunks)
                    print(f"Processado: {len(chunks)} chunks extraídos")

            except Exception as e:
                print(f"Erro ao processar o diretório {dir_path}: {e}")
                continue

        return documents

    def load_and_split_urls(self, urls: List[str]) -> List[Document]:
        """
        Loads and splits content from URLs into chunks.

        Args:
            urls: A list of URLs.

        Returns:
            A list of Document objects.
        """
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )

        for url in urls:
            if not validators.url(url):
                print(f"Aviso: URL inválida: {url}. Pulando...")
                continue

            try:
                print(f"Carregando URL: {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()

                doc = Document(page_content=text, metadata={'source': url})
                chunks = text_splitter.split_documents([doc])
                documents.extend(chunks)
                print(f"Processado: {len(chunks)} chunks extraídos")

            except requests.exceptions.RequestException as e:
                print(f"Erro ao carregar {url}: {e}")
                continue
            except Exception as e:
                print(f"Erro ao processar {url}: {e}")
                continue

        return documents

    def create_index(self, sources: List[str]) -> bool:
        """
        Cria um índice FAISS a partir de arquivos PDF e URLs.

        Args:
            sources: Uma lista de caminhos para arquivos PDF e URLs.

        Returns:
            True se o índice foi criado com sucesso, False caso contrário.
        """
        print("Criando índice FAISS...")

        pdf_paths = [source for source in sources if not validators.url(source)]
        urls = [source for source in sources if validators.url(source)]

        documents = []
        if pdf_paths:
            documents.extend(self.load_and_split_files(pdf_paths))
        if urls:
            documents.extend(self.load_and_split_urls(urls))

        self.documents = documents

        if not self.documents:
            print("Nenhum documento foi carregado. Por favor, verifique as fontes.")
            return False

        print(f"Total de chunks processados: {len(self.documents)}")

        try:
            self.vectorstore = FAISS.from_documents(
                documents=self.documents,
                embedding=self.embeddings
            )
            self.save_index()
            print("Índice FAISS criado e salvo com sucesso!")
            return True
        except Exception as e:
            print(f"Erro ao criar índice FAISS: {e}")
            return False

    def save_index(self):
        """
        Salva o índice FAISS e os metadados.
        """
        if self.vectorstore:
            self.vectorstore.save_local(self.index_path)
            metadata = [doc.metadata for doc in self.documents]
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

    def load_index(self) -> bool:
        """
        Carrega um índice FAISS existente.

        Returns:
            True se o índice foi carregado com sucesso, False caso contrário.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"Índice FAISS carregado com sucesso com {len(metadata)} documentos.")
                return True
            except Exception as e:
                print(f"Erro ao carregar índice FAISS: {e}")
                return False
        else:
            print("Índice FAISS não encontrado. Por favor, crie um índice primeiro.")
            return False

    def search(self, query: str, k: int = settings.SEARCH_K) -> List[Tuple[Document, float]]:
        """
        Realiza uma busca semântica no índice.

        Args:
            query: A consulta do usuário.
            k: O número de resultados a retornar.

        Returns:
            Uma lista de tuplas contendo o Documento e o score de similaridade.
        """
        if not self.vectorstore:
            print("Índice FAISS não carregado. Por favor, carregue ou crie um índice primeiro.")
            return []

        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"Erro durante a busca: {e}")
            return []

    def setup_azure_llm(self):
        """
        Configura o Azure OpenAI LLM.
        """
        try:
            if not settings.AZURE_OPENAI_ENDPOINT or not settings.AZURE_OPENAI_API_KEY:
                print("Variáveis de ambiente do Azure OpenAI ausentes.")
                print("Por favor, configure as variáveis de ambiente do Azure OpenAI.")
                self.use_azure_llm = False
                return

            self.llm = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                temperature=0.3,
                max_tokens=1000
            )
            print("Azure OpenAI configurado com sucesso!")
        except Exception as e:
            print(f"Erro ao configurar Azure OpenAI: {e}")
            self.use_azure_llm = False

    def setup_qa_chain(self):
        """
        Configura a cadeia de Q&A com o Azure OpenAI.

        Returns:
            True se a cadeia foi configurada com sucesso, False caso contrário.
        """
        if not self.use_azure_llm or not hasattr(self, 'llm'):
            print("Azure LLM não configurado.")
            return False

        if not self.vectorstore:
            print("Vectorstore não carregado. Por favor, carregue um índice primeiro.")
            return False

        try:
            template = """Use as seguintes partes do contexto para responder à pergunta no final.
            Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta.
            
            Contexto:
            {context}
            
            Pergunta: {question}
            
            Resposta:"""
            
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )
            
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.SEARCH_K}
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                return_source_documents=True,
                verbose=False
            )
            print("Cadeia de Q&A configurada com sucesso!")
            return True
        except Exception as e:
            print(f"Erro ao configurar a cadeia de Q&A: {e}")
            return False

    def ask_question(self, question: str) -> Tuple[str, List[Document]]:
        """
        Faz uma pergunta usando o contexto Azure OpenAI + RAG.

        Args:
            question: A pergunta do usuário.

        Returns:
            Uma tupla contendo a resposta and os documentos de origem.
        """
        if not self.qa_chain:
            return "Cadeia de Q&A não configurada. Por favor, configure o Azure OpenAI primeiro.", []

        try:
            print("Processando pergunta com Azure OpenAI...")
            result = self.qa_chain.invoke({"query": question})
            return result["result"], result["source_documents"]
        except Exception as e:
            return f"Erro ao processar a pergunta: {e}", []

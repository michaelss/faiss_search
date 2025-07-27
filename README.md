# Sistema de Busca Semântica em Documentos

Este projeto implementa um sistema de busca semântica em documentos (PDF e TXT) utilizando FAISS para criação de um índice de vetores e LangChain para orquestração. O sistema pode ser executado como uma aplicação de linha de comando (CLI) ou como uma API web com FastAPI.

## Funcionalidades

- **Indexação de Documentos:** Cria um índice vetorial a partir de arquivos PDF e TXT locais.
- **Busca por Similaridade:** Realiza uma busca semântica no índice e retorna os documentos mais relevantes para a consulta.
- **Perguntas e Respostas (Q&A):** Utiliza um modelo de linguagem da Azure OpenAI para responder perguntas com base no conteúdo dos documentos indexados (RAG).

## Opções de Execução

Existem duas formas de executar o projeto:

### 1. Aplicação de Linha de Comando (CLI)

A aplicação CLI oferece um menu interativo para executar as funcionalidades do sistema.

**Como executar:**

```bash
python main.py
```

**Opções do Menu:**

- **1. Criar novo índice:** Indexa os arquivos definidos na variável `SOURCES` no arquivo `config.py`.
- **4. Realizar busca simples (similaridade):** Busca por uma query no índice e retorna os resultados mais similares.
- **5. Fazer pergunta (Azure OpenAI + RAG):** Envia uma pergunta para o modelo de linguagem, que a responderá com base nos documentos do índice.
- **6. Sair:** Encerra a aplicação.

### 2. API Web com FastAPI

A API web expõe endpoints para interagir com o sistema de busca de forma programática.

**Como executar:**

```bash
uvicorn api:app --reload
```

A API estará disponível em `http://localhost:8000`.

**Endpoints:**

- **POST /create_index:** Cria um novo índice a partir de uma lista de fontes (caminhos de arquivos).
  - **Exemplo de corpo da requisição:**
    ```json
    {
      "sources": ["/caminho/para/meus/documentos"]
    }
    ```
- **POST /search:** Realiza uma busca por similaridade.
  - **Exemplo de corpo da requisição:**
    ```json
    {
      "query": "minha busca"
    }
    ```
- **POST /qa:** Envia uma pergunta para o sistema de Q&A.
  - **Exemplo de corpo da requisição:**
    ```json
    {
      "query": "Qual a sua pergunta?"
    }
    ```

## Configuração

- As fontes de documentos para indexação podem ser configuradas no arquivo `config.py`.
- As credenciais da Azure OpenAI devem ser configuradas como variáveis de ambiente (consulte o arquivo `.env.example` para um exemplo).

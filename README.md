# 🐕 Pakkun Project - Assistente de Código com RAG Avançado

Pakkun é um assistente inteligente de código que utiliza técnicas avançadas de RAG (Retrieval Augmented Generation) para fornecer respostas precisas sobre sua base de código. O projeto utiliza **ChromaDB** como banco de dados vetorial local com **coleções individuais por arquivo** e **Streamlit** para a interface interativa.

## 📊 Funcionalidades

- **Indexação por Arquivo**: Cada arquivo tem sua própria coleção para buscas mais precisas e contextualizadas
- **Parsers Específicos por Linguagem**: Processamento especializado para Python e JavaScript/TypeScript
- **Chunking Otimizado**: Chunks maiores e com sobreposição para melhor contexto e compreensão
- **Busca Semântica Avançada**: Múltiplas estratégias de busca com priorização inteligente
- **Extração Estrutural**: Detecção sofisticada de classes, funções, componentes React e hooks
- **Interface Intuitiva**: Interface de chat amigável com formatação melhorada e visualização de raciocínio

## 📋 Pré-requisitos

- **Python** 3.8+
- **Streamlit** (https://streamlit.io/)
- **ChromaDB** para armazenamento de vetores
- **Bibliotecas listadas em** `requirements.txt`
- **Chave API para o Groq** (ou outro LLM compatível)

## 🚀 Instalação

### 1️⃣ Clonar o repositório:
```sh
git clone <URL_DO_REPOSITÓRIO>
cd <NOME_DO_PROJETO>
```

### 2️⃣ Configurar ambiente virtual:
```sh
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3️⃣ Instalar dependências:
```sh
pip install -r requirements.txt
```

### 4️⃣ Configurar variáveis de ambiente:
Crie um arquivo `.env` na raiz do projeto:
```ini
GROQ_API_KEY=your_groq_api_key_here
# Outras variáveis se necessário
```

## 🔍 Indexação dos Dados

Antes de usar o assistente, é necessário indexar sua base de código:

```sh
python core/code_indexer.py --folder <caminho_da_pasta>
```

Este processo aprimorado:
- Lê todos os arquivos de código no diretório especificado
- Cria uma coleção ChromaDB individual para cada arquivo
- Utiliza parsers específicos para diferentes linguagens (Python, JavaScript/TypeScript)
- Extrai estruturas complexas como classes, métodos, decoradores, componentes React e hooks
- Gera chunks maiores (2500 caracteres) com alta sobreposição (300 caracteres)
- Mantém um mapeamento JSON de arquivos para coleções para consulta rápida
- Armazena metadados ricos sobre cada documento para filtragem avançada

## 🤖 Executando o Assistente

Após a indexação, execute:
```sh
streamlit run core/app.py
```

Seu navegador abrirá automaticamente com a interface do Pakkun, onde você pode:
- Fazer perguntas sobre sua base de código
- Mencionar arquivos específicos para obter informações detalhadas
- Ver explicações técnicas com formatação adequada
- Acompanhar o raciocínio detalhado do assistente (opcional)

## 🧠 Recursos Avançados

- **MultiCollectionRAG**: Sistema que busca em coleções individuais para resultados mais precisos
- **Detecção Aprimorada de Arquivos**: Reconhece menções a arquivos em consultas, mesmo parciais
- **Estratégias de Busca Adaptativas**: Combina busca específica por arquivo com busca global
- **Resposta Inteligente sobre Arquivos**: Detecta consultas sobre listagem de arquivos e fornece sumários
- **Deduplicação e Reranking**: Algoritmos que priorizam, deduplicam e diversificam resultados
- **Extração Estrutural por Linguagem**: Analisa código com regras específicas para cada linguagem
- **Memória de Conversa**: Mantém o contexto da conversa para respostas mais precisas
- **Visualização de Raciocínio**: Formato <think>...</think> para explicar o processo de pensamento

## 🔧 Personalização

Você pode personalizar vários aspectos do sistema:
- Modelos de embedding no arquivo `code_indexer.py`
- Modelo LLM usado para respostas em `app.py`
- Parâmetros de busca e contexto no `RAGAssistant`
- Aparência da interface através das configurações de tema

## 📂 Estrutura do Projeto

```
.
├── core/
│   ├── code_indexer.py   # Sistema de indexação avançado
│   └── app.py            # Interface Streamlit e sistema RAG
├── chroma_db/            # Banco de dados vetorial local (gerado)
├── requirements.txt      # Dependências do projeto
├── .env                  # Variáveis de ambiente (não commitado)
└── README.md             # Este arquivo
```

## 🤝 Contribuições

Contribuições são bem-vindas! Se você tiver sugestões para melhorar o sistema:
1. Faça um fork do projeto
2. Crie uma branch para sua funcionalidade (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas alterações (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📝 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## 📞 Contato

Para questões e sugestões, por favor abra uma issue no repositório.


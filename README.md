# 🐕 Pakkun Project - Assistente de Código com RAG

Pakkun é um assistente inteligente de código que utiliza técnicas avançadas de RAG (Retrieval Augmented Generation) para fornecer respostas precisas sobre sua base de código. O projeto utiliza **ChromaDB** como banco de dados vetorial local e **Streamlit** para a interface interativa.

## 📊 Funcionalidades

- **Indexação Inteligente**: Análise e indexação de códigos com extração de estruturas específicas como funções, classes e componentes
- **Busca Semântica**: Recuperação de informações baseada em significado usando embeddings, não apenas correspondência de palavras-chave
- **Múltiplas Estratégias de Recuperação**: Combina diferentes métodos para encontrar o conteúdo mais relevante
- **Interface Intuitiva**: Interface de chat amigável para interagir com o assistente

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

Este processo:
- Lê todos os arquivos de código no diretório especificado
- Extrai estruturas importantes como classes e funções
- Divide os documentos em chunks para busca semântica
- Gera embeddings usando o modelo sentence-transformers
- Armazena tudo no banco ChromaDB local

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

- **Deduplicação e Reranking**: Algoritmos que priorizam e diversificam resultados
- **Extração Inteligente**: Analisa diferentes linguagens de programação e extrai estruturas específicas
- **Memória de Conversa**: Mantém o contexto da conversa para respostas mais precisas
- **Filtros Contextuais**: Busca otimizada baseada no tipo de pergunta

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


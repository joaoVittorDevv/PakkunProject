# Pakkun Project

Este projeto consiste em um sistema de indexação de código e uma interface de chat que utiliza os dados indexados para fornecer respostas baseadas em similaridade de embeddings. O projeto utiliza o **Pinecone** como banco de dados vetorial e **Streamlit** para a interface interativa.

---

## 📌 Pré-requisitos

Certifique-se de ter os seguintes requisitos antes de iniciar:

- **Python** 3.8+
- **Conta e chave de API** do Pinecone
- **Streamlit** (https://streamlit.io/)
- **Bibliotecas listadas em** `requirements.txt`

---

## 🚀 Instalação

### 1️⃣ Clonar o repositório e acessar a pasta do projeto:
```sh
# Clone o repositório
git clone <URL_DO_REPOSITÓRIO>

# Entre no diretório do projeto
cd <NOME_DO_PROJETO>
```

### 2️⃣ Criar e ativar um ambiente virtual (opcional, mas recomendado):
```sh
# Criar um ambiente virtual
python -m venv venv

# Ativar o ambiente virtual (Linux/Mac)
source venv/bin/activate

# No Windows
venv\Scripts\activate
```

### 3️⃣ Instalar as dependências:
```sh
pip install -r requirements.txt
```

### 4️⃣ Configurar as variáveis de ambiente:
Crie um arquivo `.env` na raiz do projeto e defina as variáveis necessárias, por exemplo:
```ini
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
# Outras variáveis necessárias...
```

---

## 🔍 Indexação dos Dados

Antes de rodar o app, é necessário indexar os arquivos de código.

### 📌 Passo 1: Executar o script de indexação
Utilize o comando abaixo, substituindo `<caminho da pasta>` pelo diretório contendo os arquivos que deseja indexar:
```sh
python core/code_indexer.py --folder <caminho da pasta>
```

🔹 Esse comando irá:
- Ler os arquivos de código no diretório especificado;
- Gerar os embeddings dos documentos utilizando o modelo configurado;
- Indexar os dados no Pinecone.

---

## 🎯 Executando o Projeto

Após a indexação dos dados, inicie a interface do chat com o comando:
```sh
streamlit run core/app.py
```
O Streamlit abrirá uma nova janela ou aba no navegador com a interface do Pakkun, permitindo interações e consultas baseadas nos dados indexados.

---

## 📂 Estrutura do Projeto

```
.
├── core
│   ├── code_indexer.py   # Script para indexação dos dados de código
│   └── app.py            # Aplicação Streamlit (interface do chat)
├── requirements.txt      # Dependências do projeto
├── .env                  # Variáveis de ambiente (não commitado)
└── README.md             # Este arquivo
```

---

## 🔧 Personalização e Contribuições

### 🔄 Refino do Código
Este projeto está em fase de desenvolvimento. Caso tenha sugestões ou melhorias, fique à vontade para contribuir ou abrir uma issue no repositório.

### ⚙️ Customização
Se desejar modificar a indexação, modelo de embeddings ou interface, você pode ajustar os parâmetros conforme sua necessidade nos scripts do diretório `core`.



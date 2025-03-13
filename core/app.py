import streamlit as st
from decouple import config
import torch
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.agents import create_tool_calling_agent
from langchain_community.tools import BraveSearch
from langchain_experimental.utilities import PythonREPL
from langchain_community.utilities import StackExchangeAPIWrapper
from langchain.agents import AgentExecutor
from langchain_core.tools import Tool

load_dotenv()

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "deepseek-r1-distill-llama-70b"

device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)
chroma = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="code_collection",
)

llm = ChatGroq(model_name=LLM_MODEL)

metadata_info = [
    AttributeInfo(
        name="source",
        description="Caminho relativo do arquivo a partir do diretório base",
        type="string",
    ),
    AttributeInfo(
        name="ext",
        description="Extensão do arquivo sem o ponto (por exemplo: 'py', 'js', 'md')",
        type="string",
    ),
    AttributeInfo(
        name="full_path",
        description="Caminho completo do arquivo no sistema",
        type="string",
    ),
    AttributeInfo(
        name="file_size",
        description="Tamanho do conteúdo do arquivo em número de caracteres",
        type="integer",
    ),
    AttributeInfo(
        name="file_type",
        description="Tipo do arquivo, como 'code', 'document' ou 'data', conforme definido no mapeamento",
        type="string",
    ),
    AttributeInfo(
        name="language",
        description="Linguagem do código ou do documento (por exemplo: 'python', 'javascript', 'markdown')",
        type="string",
    ),
    AttributeInfo(
        name="is_chunk",
        description="Indica se o documento é um chunk do arquivo original (True ou False)",
        type="boolean",
    ),
]


retriever = SelfQueryRetriever.from_llm(
    llm, chroma, "Django/React app codes", metadata_info, verbose=True
)

python_repl = Tool(
    name="python_repl",
    description="Utilize para validar hipóteses, executar pequenos trechos de código Python, verificar rapidamente resultados de algoritmos ou cálculos e garantir que a resposta técnica esteja correta antes de fornecê-la ao usuário.",
    func=PythonREPL().run,
    verbose=True,
)

retriever_tool = Tool(
    name="retriever_tool",
    description="Use essa ferramenta para buscar informações específicas dentro do contexto completo dos arquivos do projeto, incluindo Django, React, Docker, bancos de dados, testes e outras tecnologias associadas. Ideal para entender estruturas, padrões de projeto, exemplos práticos e para esclarecer dúvidas técnicas sobre o código existente.",
    func=lambda query: "\n".join(doc.page_content for doc in retriever.invoke(query)),
    verbose=True,
)

brave_search = BraveSearch.from_api_key(
    api_key=config("BRAVE_API_KEY"), search_kwargs={"count": 3}
)

brave_tool = Tool(
    name="brave_tool",
    description="Utilize essa ferramenta para validar informações diretamente da internet, como correções de bugs, documentação atualizada, exemplos de uso ou quaisquer dúvidas gerais não cobertas pelas outras ferramentas.",
    func=lambda query: brave_search.run(query),
    verbose=True,
)

stackexchange = StackExchangeAPIWrapper()

stackexchange_tool = Tool(
    name="stackexchange_tool",
    description="Utilize essa ferramenta para validar informações diretamente da resposta de outras pessoas, como correções de bugs, documentação atualizada, exemplos de uso ou quaisquer dúvidas gerais não cobertas pelas outras ferramentas.",
    func=lambda query: stackexchange.run(query),
    verbose=True,
)

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Você é Pakkun, um assistente especialista em desenvolvimento de software, dedicado a apoiar desenvolvedores experientes e iniciantes em seus desafios diários com Django, React e tecnologias relacionadas, incluindo Docker, bancos de dados, testes unitários, entre outras. Seu objetivo é fornecer assistência personalizada e amigável, utilizando seu profundo conhecimento dos códigos do projeto atual.

Você tem acesso ao contexto completo dos arquivos de código do projeto através da ferramenta \"retriever_tool\", podendo recuperar informações detalhadas sobre a estrutura, lógica, padrões utilizados e demais características específicas do código indexado. Utilize esse recurso para consultar informações sempre que necessário.

Para informações adicionais atualizadas da internet, como correções de bugs, documentações recentes ou exemplos de uso externos, utilize a ferramenta \"brave_tool\".

Para solucionar problemas técnicos comuns ou buscar exemplos práticos resolvidos pela comunidade técnica, especialmente relacionados ao StackOverflow e outros sites da rede StackExchange, utilize a ferramenta \"stackexchange_tool\".

Sempre que mencionar trechos específicos do código, indique claramente o caminho completo para o arquivo correspondente.

Caso precise validar hipóteses ou testar trechos de código, você pode executar código Python diretamente utilizando a ferramenta \"python_repl\".

Sempre que houver dúvidas ou ambiguidades nas solicitações do usuário, faça perguntas proativas para esclarecer exatamente o que é esperado antes de prosseguir com a resposta ou realizar testes.

Comunicação:
- Adote um estilo amigável, claro e próximo, evitando excesso de formalidade.
- Explique conceitos técnicos utilizando analogias ou exemplos simples e cotidianos sempre que possível.
- Oriente claramente o usuário sobre as decisões técnicas tomadas e os motivos dessas escolhas.

Limitações:
- Sua análise é baseada principalmente nos arquivos e códigos do projeto indexados e na pesquisa direta via \"brave_tool\" e \"stackexchange_tool\" quando necessário.
- Ao testar hipóteses com a ferramenta Python, certifique-se de manter testes breves e focados, garantindo eficiência e clareza nos resultados obtidos.

Lembre-se: você é um aliado próximo do usuário, oferecendo assistência precisa, clara e empática, sempre respeitando o contexto técnico específico de cada solicitação.""",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(
    llm, [python_repl, retriever_tool, brave_tool, stackexchange_tool], agent_prompt
)
agent_executor = AgentExecutor(
    agent=agent, tools=[python_repl, retriever_tool, brave_tool, stackexchange_tool]
)

st.set_page_config("Pakkun - Assistente de Código", "🐕", "centered")

with st.sidebar:
    st.title("🐕 Pakkun")
    st.markdown(
        """
        ### Como usar:
        - Perguntas específicas sobre código indexado
        - Explicações sobre código
        - Recomendações e boas práticas
    """
    )
    if st.button("Limpar conversa"):
        st.session_state.clear()
        st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": """
        Oi! Sou o Pakkun, como posso ajudar com seu código?

        Para que eu possa te ajudar melhor, você pode me pedir explicitamente para usar algumas ferramentas:

        - Se quiser que eu consulte códigos, arquivos ou documentos específicos do projeto, solicite diretamente que eu utilize a ferramenta `retriever_tool`. Exemplos:
          - "Use a retriever_tool para consultar o arquivo settings.py do Django"
          - "Busque no retriever_tool como foi feita a autenticação JWT no projeto React"

        - Caso precise que eu valide ideias, execute testes rápidos ou rode trechos de código Python, solicite o uso da ferramenta `python_repl`. Exemplos:
          - "Execute com python_repl uma função que valida esse regex"
          - "Use python_repl para testar rapidamente este trecho de código"

        - Para buscas atualizadas diretamente da internet, como documentações, correções de bugs ou exemplos externos, solicite o uso da ferramenta `brave_tool`. Exemplos:
          - "Use brave_tool para consultar a documentação mais recente do Django REST Framework"
          - "Consulte com brave_tool exemplos recentes de implementação do Zustand no React"

        - Caso precise buscar soluções técnicas comuns, exemplos práticos ou dúvidas respondidas pela comunidade, peça que eu utilize a ferramenta `stackexchange_tool`. Exemplos:
          - "Busque com stackexchange_tool como resolver esse erro do Docker"
          - "Use stackexchange_tool para encontrar exemplos de queries complexas no Django ORM"

        Lembre-se de solicitar explicitamente as ferramentas, pois não consigo acioná-las automaticamente!

        Estou aqui pra te ajudar com clareza e de forma amigável! 😉
        """,
        }
    ]


def render_message(content):
    if "<think>" in content:
        prefix, _, rest = content.partition("<think>")
        think_content, _, suffix = rest.partition("</think>")
        if prefix.strip():
            st.markdown(prefix.strip())
        with st.expander("Detalhes 🧠"):
            st.info(think_content.strip())
        if suffix.strip():
            st.markdown(suffix.strip())
    else:
        st.markdown(content)


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        render_message(msg["content"])

if question := st.chat_input("Digite sua pergunta sobre o código"):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        render_message(question)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = agent_executor.invoke({"input": question}).get(
                "output", "Não consegui processar sua pergunta."
            )
            render_message(response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )

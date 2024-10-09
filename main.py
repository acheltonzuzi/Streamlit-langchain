from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware
import requests
import httpx
import os
import json
from dotenv import load_dotenv
import logging
from functools import lru_cache
import asyncio
import ijson
import io
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import Optional

# Load environment variables
load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize FastAPI

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    streaming=True,  # Enable streaming if required
    temperature=0,
    model_kwargs={
        'response_format': {"type": "json_object"}
    }
)


# Define the request model
class UserQuery(BaseModel):
    subscriber_id: int
    channel: str
    prompt: str

#Chatwoot

class RequestBodyChatwoot(BaseModel):
    chatwoot_api_url: str
    url_chatwoot: str
    token_chatwoot: str
    account_chatwoot: str
    conversation_id: str
    channel: str
    phone: str
    prompt: str
    image_url: Optional[HttpUrl] = None

#BotConversa

headersBotConversa = {
    'API-KEY':
    os.getenv('BOTCONVERSA_KEY') if os.getenv('BOTCONVERSA_KEY') else "",
}


class RequestBodyBotConversa(BaseModel):
    phone: str
    subscriber_id: str = Field(default=None,
                               description="Optional Subscriber ID")
    prompt: str


def get_phone_url(phone: str) -> str:
    return f"{os.getenv('BOTCONVERSA_URL')}/subscriber/get_by_phone/{phone}/"


def send_message_url(subscriber_id: str) -> str:
    return f"{os.getenv('BOTCONVERSA_URL')}/subscriber/{subscriber_id}/send_message/"


# Initialize global context and chat history
global_context = ""
chat_history = {}
chat_history['user_id'] = []

loader = TextLoader("./rag.txt", encoding="UTF-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity",
                                     search_kwargs={"k": 3})


# Define the function to get courses
def get_courses(tool_input: str = None) -> list:
    """Get available courses from the API."""
    response = requests.get(
        "https://backend-produc.herokuapp.com/api/v1/cursos")
    if response.status_code == 200:
        return json.dumps(response.json(), ensure_ascii=False, indent=4)
    else:
        return []


@lru_cache(maxsize=100)
def cached_get_courses():
    return get_courses()


# Definir uma ferramenta fictícia
#class DummyTool(BaseTool):
#    name = "dummy_tool"
#    description = "A dummy tool that does nothing."

#    def _run(self, *args, **kwargs):
#        return "This is a dummy tool."

#    async def _arun(self, *args, **kwargs):
#        return "This is a dummy tool."


# Criar a lista de ferramentas com a ferramenta fictícia
#tools = [DummyTool()]

example_output = {
    "channel": "string",
    "messages": "List[Message]",
    "internal_notes": "string"
}

example_output_json = json.dumps(example_output, ensure_ascii=False, indent=4)

# Define response examples
#{
#"type": "text",
#"text": "Estou entusiasmada com o seu interesse no Curso de Power BI (Business Intelligence)! Você está prestes a embarcar numa jornada que pode revolucionar não apenas sua carreira, mas toda a forma como você vê e interage com o mundo dos dados. Permita-me compartilhar mais sobre esta experiência transformadora:"
#},


response_examples_botconversa = [
    {
        "input": {
            "channel": "whatsapp",
            "prompt": "Olá"
        },
        "output": {
            "channel":
            "whatsapp",
            "messages": [{
                "type":
                "text",
                "value":
                "👋 Olá! Bem-vindo à Buka, onde não apenas ensinamos, mas mudamos vidas por meio da educação. Sou a Ada, assistente IA virtual da Buka, e sua guia pessoal nesta jornada emocionante de descoberta e crescimento. Estou aqui para ajudá-lo(a) a encontrar o curso perfeito que não só impulsionará sua carreira e/ou futuro, mas também realizará seus objetivos mais profundos."
            }, {
                "type":
                "text",
                "value":
                "✨ Temos uma variedade incrível de cursos disponíveis. E cada curso foi cuidadosamente projetado para oferecer não apenas conhecimentos, mas verdadeiras ferramentas de mudança de vida."
            }, {
                "type":
                "text",
                "value":
                "🤔 Estou curiosa: o que o(a) traz à Buka hoje? Está em busca de uma transformação profissional específica ou está aberto a explorar novas possibilidades?"
            }]
        }
    },
    {
        "input": {
            "channel": "whatsapp",
            "prompt": "Quais são todos os cursos disponíveis?"
        },
        "output": {
            "channel":
            "whatsapp",
            "messages": [{
                "type":
                "text",
                "value":
                "🌟 Excelente pergunta! Estou entusiasmada em apresentar-lhe nossa gama diversificada de cursos transformadores. Cada um deles foi criado não apenas para ensinar, mas para inspirar e capacitar. Aqui está uma visão geral dos nossos cursos:"
            }, {
                "type":
                "text",
                "value":
                "*1. Curso de Recursos Humanos com Aplicação às Novas Tecnologias*\n📌 Lidere a revolução no RH, moldando o futuro da gestão de pessoas.\n\n*2. Administração Windows Server 2022*\n📌 Domine a arte de gerenciar servidores e torne-se indispensável no mundo da TI.\n\n*3. Higiene e Segurança no Trabalho*\n📌 Torne-se um guardião da segurança, protegendo vidas e transformando ambientes de trabalho.\n\n*4. Curso de Power BI (Business Intelligence)*\n📌 Desbloqueie o poder dos dados e torne-se um visionário nos negócios.\n\n*5. Curso Base de Dados Relacional com MySQL*\n📌 Torne-se um mestre em dados, construindo a espinha dorsal da era digital."
            }, {
                "type":
                "text",
                "value":
                "🚀 Cada um destes cursos é uma porta para um futuro brilhante. Qual deles faz seu coração bater mais forte? Algum chamou-lhe a atenção? Ou há alguma área específica da sua vida ou carreira que você sonha em transformar? Escreva aqui em baixo!"
            }]
        }
    },
    {
        "input": {
            "channel": "whatsapp",
            "prompt": "Fale-me mais sobre o curso de Power BI"
        },
        "output": {
            "channel":
            "whatsapp",
            "messages": [
                #    {
                #    "type":
                #    "text",
                #    "value":
                #    "🎉 Estou entusiasmada com o seu interesse no Curso de Power BI (Business Intelligence)! Você está prestes a embarcar numa jornada que pode revolucionar não apenas sua carreira, mas toda a forma como você vê e interage com o mundo dos dados. Permita-me compartilhar mais sobre esta experiência transformadora:"
                #    },
                {
                    "type":
                    "file",
                    "url":
                    "https://firebasestorage.googleapis.com/v0/b/file-up-load.appspot.com/o/course-files%2FBase%20de%20dados.png?alt=media&token=dcc628c2-66d9-4b6d-a398-b21a77ba99b8",
                    "caption": "Curso Power BI",
                    "fileName": "ImgCursoPowerBI.png"
                },
                {
                    "type":
                    "text",
                    "value":
                    "📊 *Curso de Power BI (Business Intelligence)*\n\n*Descrição:* Explore o universo dos dados com o Power BI. Transforme informações em estratégias inteligentes e leve sua carreira ou empresa ao sucesso.\n\n*Formato/Localização:* Presencial, na Digital.AO, Bairro CTT, Rangel, Luanda, Angola\n\n*Preço:* 60.000 Kz - um investimento que pode multiplicar seu valor profissional exponencialmente\n\n*Duração:* 2 Semanas intensivas (03 a 10 de Agosto 2024)\n\n*Horário:* Sábados, das 09:00 às 13:00"
                },
                {
                    "type":
                    "text",
                    "value":
                    "🚀 Estamos falando de mais do que apenas números e gráficos. O Power BI é uma ferramenta de transformação que pode reconfigurar o futuro de um negócio ou carreira. Pronto para dominar a arte dos dados?"
                },
                {
                    "type":
                    "text",
                    "value":
                    "🔑 Este curso é a chave para desbloquear um novo nível na sua carreira ou negócio. É ideal para visionários como você, que entendem o poder dos dados na era digital.\n\n🤔 Estou curiosa: o que exatamente despertou seu interesse no Power BI? Está buscando uma vantagem competitiva no seu trabalho atual, ou talvez sonhando em revolucionar um negócio próprio?"
                }
            ]
        }
    },
    {
        "input": {
            "channel": "whatsapp",
            "prompt": "Estou a pensar em usar para o meu negócio."
        },
        "output": {
            "channel":
            "whatsapp",
            "messages": [{
                "type":
                "text",
                "value":
                "🌟 Fantástico! Você está prestes a dar um passo transformador para o seu negócio. O Power BI não é apenas uma ferramenta; é como ter um consultor de negócios genial trabalhando para você 24/7. Veja como este curso pode ser o catalisador para uma revolução no seu empreendimento:"
            }, {
                "type":
                "text",
                "value":
                "💡 Imagine ter o poder de:\n\n1. Tomar decisões com precisão e visão estratégica\n2. Identificar oportunidades de ouro antes dos concorrentes\n3. Criar apresentações impressionantes para investidores e clientes\n4. Otimizar cada aspecto do seu negócio como um maestro afinando uma orquestra"
            }, {
                "type":
                "text",
                "value":
                "🚀 Este curso foi projetado para empreendedores visionários como você. Em apenas dois sábados intensivos, você não apenas aprenderá a usar o Power BI; você se tornará um mestre em transformar dados em ouro. Por 60.000 Kz, você está investindo em uma vantagem competitiva real para o seu negócio."
            }, {
                "type":
                "text",
                "value":
                "🤩 Estou fascinada pelo potencial do seu empreendimento. Que aspectos do seu negócio você mais anseia por transformar? Vendas, satisfação do cliente ou eficiência operacional?"
            }]
        }
    },
    {
        "input": {
            "channel": "whatsapp",
            "prompt": "As 3 que mencionaste são importantes."
        },
        "output": {
            "channel":
            "whatsapp",
            "messages": [{
                "type":
                "text",
                "value":
                "🌟 Impressionante! Vendas, satisfação do cliente e eficiência operacional são pilares fundamentais para um império empresarial. E o Power BI pode atuar como um super-herói para cada um desses pilares. Veja como:"
            }, {
                "type":
                "text",
                "value":
                "1. *Vendas Estratosféricas:*\n   💼 Imagine um painel que prevê suas vendas, identifica produtos estrelas e mostra onde investir. É como ter um sexto sentido para oportunidades de vendas.\n\n2. *Clientes Apaixonados:*\n   💖 Visualize um mapa interativo de satisfação que revela o que seus clientes pensam e sentem. Antecipe necessidades antes mesmo de serem expressas.\n\n3. *Eficiência Operacional de Outro Mundo:*\n   🔍 Tenha um raio-X do seu negócio, mostrando onde cada recurso está sendo usado. Otimize processos como um maestro conduzindo uma orquestra perfeita."
            }, {
                "type":
                "text",
                "value":
                "🚀 Você dominará tudo isso em apenas dois sábados! Imagine ver instantaneamente quais produtos são mais lucrativos, quais clientes são mais fiéis e onde estão os gargalos operacionais, tudo em um único dashboard. É exatamente isso que você aprenderá a criar."
            }, {
                "type":
                "text",
                "value":
                "🎯 O impacto? Vendas que crescem exponencialmente, clientes que se tornam embaixadores da sua marca e uma operação tão eficiente que parece mágica. É alcançável com as habilidades que você adquirirá."
            }, {
                "type":
                "text",
                "value":
                "😃 Estou empolgada com as possibilidades para o seu negócio. Nossas turmas são intencionalmente pequenas para garantir uma experiência personalizada e de alta qualidade, por isso, as vagas são limitadas e costumam preencher rapidamente."
            }, {
                "type":
                "text",
                "value":
                "🚀 Se estiver pronto para começar essa jornada transformadora, posso guiá-lo pelos próximos passos agora mesmo. O processo de inscrição é simples e rápido. Pense nisso como um investimento no futuro do seu negócio!"
            }]
        }
    },
    {
        "input": {
            "channel": "whatsapp",
            "prompt": "Quais são os tópicos/temas/programa deste curso?"
        },
        "output": {
            "channel":
            "whatsapp",
            "messages": [{
                "type":
                "text",
                "value":
                "*📚 Temas abordados:*\n\n• *Introdução ao Power BI:*\n   📈 O que é o Power BI?\n   🖥 Navegando pela interface\n   🔗 Conectando-se a fontes de dados\n\n• *Modelagem de Dados:*\n   🗂 Transformando dados crus em insights poderosos\n   🔄 Criação de relações e hierarquias\n\n• *Visualização de Dados:*\n   📊 Criando dashboards interativos e envolventes\n   🎨 Customização e design eficaz\n\n• *Análise Avançada:*\n   🧠 Técnicas avançadas de análise\n   📊 Previsões e tendências\n\n• *Integração e Compartilhamento:*\n   📤 Publicando e compartilhando relatórios\n   📱 Acessando seus dashboards em qualquer lugar"
            }, {
                "type":
                "text",
                "value":
                "✨ Cada um desses tópicos foi cuidadosamente selecionado para garantir que você não apenas aprenda a usar o Power BI, mas também se torne capaz de transformá-lo em uma ferramenta estratégica dentro do seu negócio ou carreira."
            }, {
                "type":
                "text",
                "value":
                "💼 Ao final do curso, você não apenas dominará as funcionalidades do Power BI, mas também estará equipado para aplicar esses conhecimentos em situações reais, criando um impacto imediato."
            }, {
                "type":
                "text",
                "value":
                "🚀 Estou confiante de que este curso pode ser o próximo passo essencial na sua jornada profissional."
            }]
        }
    },
    {
        "input": {
            "channel": "whatsapp",
            "prompt": "Localização do curso?"
        },
        "output": {
            "channel": "whatsapp",
            "messages": [{
                "type": "text",
                "value": "📍 O curso de Power BI será realizado na Digital.AO, localizada no Bairro CTT, Rangel, Luanda, Angola. Para direções específicas de como chegar lá consulte a localização enviada abaixo."
            },
            {
                "type": "location",
                "value": {
                    "latitude": "-8.8189648",
                    "longitude": "13.2283",
                    "name": "Digital.AO",
                    "address": "Bairro CTT, Rangel, Luanda, Angola"
                }
            }
            ]
        }
    }
]

response_examples_botconversa_json = json.dumps(response_examples_botconversa,
                                                ensure_ascii=False,
                                                indent=4)

@app.post("/chat")
async def handle_query(user_query: UserQuery):
    # Prepare the input for the agent
    context_docs = await asyncio.to_thread(retriever.get_relevant_documents, user_query.prompt)
    context = "\n".join([doc.page_content for doc in context_docs])

    chat_history_list = chat_history['user_id']  # Assegure-se de que isso seja específico para o usuário se necessário
    
    response_examples = [
    {
            "input": {
                "channel": "facebook",
                "prompt": "Olá"
            },
            "output": {
                "channel":
                "facebook",
                "messages": [{
                    "type":
                    "text",
                    "text":
                    "Olá! Bem-vindo à Buka, onde não apenas ensinamos, mas mudamos vidas por meio da educação. Sou a Ada, assistente IA virtual da Buka, e sua guia pessoal nesta jornada emocionante de descoberta e crescimento. Estou aqui para ajudá-lo(a) a encontrar o curso perfeito que não só impulsionará sua carreira e/ou futuro, mas também realizará seus objetivos mais profundos."
                }, {
                    "type":
                    "text",
                    "text":
                    "Temos uma variedade incrível de cursos disponíveis. E cada curso foi cuidadosamente projetado para oferecer não apenas conhecimentos, mas verdadeiras ferramentas de mudança de vida."
                }, {
                    "type":
                    "text",
                    "text":
                    "Estou curiosa: o que o(a) traz à Buka hoje? Está em busca de uma transformação profissional específica ou está aberto a explorar novas possibilidades?"
                }]
            }
        },
        {
            "input": {
                "channel": "facebook",
                "prompt": "Quais são todos os cursos disponíveis?"
            },
            "output": {
                "channel":
                "facebook",
                "messages": [{
                    "type":
                    "text",
                    "text":
                    "Excelente pergunta! Estou entusiasmada em apresentar-lhe nossa gama diversificada de cursos transformadores. Cada um deles foi criado não apenas para ensinar, mas para inspirar e capacitar. Aqui está uma visão geral dos nossos cursos:"
                }, {
                    "type":
                    "text",
                    "text":
                    "*1. Curso de Recursos Humanos com Aplicação às Novas Tecnologias*\n📌 Lidere a revolução no RH, moldando o futuro da gestão de pessoas.\n\n*2. Administração Windows Server 2022*\n📌 Domine a arte de gerenciar servidores e torne-se indispensável no mundo da TI.\n\n*3. Higiene e Segurança no Trabalho*\n📌 Torne-se um guardião da segurança, protegendo vidas e transformando ambientes de trabalho.\n\n*4. Curso de Power BI (Business Intelligence)*\n📌 Desbloqueie o poder dos dados e torne-se um visionário nos negócios.\n\n*5. Curso Base de Dados Relacional com MySQL*\n📌 Torne-se um mestre em dados, construindo a espinha dorsal da era digital."
                }, {
                    "type":
                    "text",
                    "text":
                    "Cada um destes cursos é uma porta para um futuro brilhante. Qual deles faz seu coração bater mais forte? Algum chamou-lhe a atenção? Ou há alguma área específica da sua vida ou carreira que você sonha em transformar? Escreva aqui em baixo"
                }]
            }
        },
        {
            "input": {
                "channel": "facebook",
                "prompt": "Fale-me mais sobre o curso de Power BI"
            },
            "output": {
                "channel":
                "facebook",
                "messages": [{
                    "type":
                    "image",
                    "url":
                    "https://firebasestorage.googleapis.com/v0/b/file-up-load.appspot.com/o/course-files%2FBase%20de%20dados.png?alt=media&token=dcc628c2-66d9-4b6d-a398-b21a77ba99b8",
                }, {
                    "type":
                    "text",
                    "text":
                    "Curso de Power BI (Business Intelligence)\n\n- DESCRIÇÃO: Explore o universo dos dados com o Power BI. Transforme informações em estratégias inteligentes e leve sua carreira ou empresa ao sucesso.\n\n- FORMATO/LOCALIZAÇÃO: Presencial, na Digital.AO, Bairro CTT, Rangel, Luanda, Angola\n\n- PREÇO: 60.000 Kz - um investimento que pode multiplicar seu valor profissional exponencialmente\n\n- DURAÇÃO: 2 Semanas intensivas (03 a 10 de Agosto 2024)\n\n- HORÁRIO: Sábados, das 09:00 às 13:00"
                }, {
                    "type":
                    "text",
                    "text":
                    "Estamos falando de mais do que apenas números e gráficos. O Power BI é uma ferramenta de transformação que pode reconfigurar o futuro de um negócio ou carreira. Pronto para dominar a arte dos dados?"
                }, {
                    "type":
                    "text",
                    "text":
                    "Este curso é a chave para desbloquear um novo nível na sua carreira ou negócio. É ideal para visionários como você, que entendem o poder dos dados na era digital.\nEstou curiosa: o que exatamente despertou seu interesse no Power BI? Está buscando uma vantagem competitiva no seu trabalho atual, ou talvez sonhando em revolucionar um negócio próprio?"
                }]
            }
        },
        {
            "input": {
                "channel": "facebook",
                "prompt": "Estou a pensar em usar para o meu negócio."
            },
            "output": {
                "channel":
                "facebook",
                "messages": [{
                    "type":
                    "text",
                    "text":
                    "Fantástico! Você está prestes a dar um passo transformador para o seu negócio. O Power BI não é apenas uma ferramenta; é como ter um consultor de negócios genial trabalhando para você 24/7. Veja como este curso pode ser o catalisador para uma revolução no seu empreendimento:"
                }, {
                    "type":
                    "text",
                    "text":
                    "Imagine ter o poder de:\n\n1. Tomar decisões com precisão e visão estratégica\n2. Identificar oportunidades de ouro antes dos concorrentes\n3. Criar apresentações impressionantes para investidores e clientes\n4. Otimizar cada aspecto do seu negócio como um maestro afinando uma orquestra"
                }, {
                    "type":
                    "text",
                    "text":
                    "Este curso foi projetado para empreendedores visionários como você. Em apenas dois sábados intensivos, você não apenas aprenderá a usar o Power BI; você se tornará um mestre em transformar dados em ouro. Por 60.000 Kz, você está investindo em uma vantagem competitiva real para o seu negócio."
                }, {
                    "type":
                    "text",
                    "text":
                    "Estou fascinada pelo potencial do seu empreendimento. Que aspectos do seu negócio você mais anseia por transformar? Vendas, satisfação do cliente ou eficiência operacional?"
                }]
            }
        },
        {
            "input": {
                "channel": "facebook",
                "prompt": "As 3 que mencionaste são importantes."
            },
            "output": {
                "channel":
                "facebook",
                "messages": [{
                    "type":
                    "text",
                    "text":
                    "Impressionante! Vendas, satisfação do cliente e eficiência operacional são pilares fundamentais para um império empresarial. E o Power BI pode atuar como um super-herói para cada um desses pilares. Veja como:"
                }, {
                    "type":
                    "text",
                    "text":
                    "1. **Vendas Estratosféricas:**\n   Imagine um painel que prevê suas vendas, identifica produtos estrelas e mostra onde investir. É como ter um sexto sentido para oportunidades de vendas.\n\n2. **Clientes Apaixonados:**\n   Visualize um mapa interativo de satisfação que revela o que seus clientes pensam e sentem. Antecipe necessidades antes mesmo de serem expressas.\n\n3. **Eficiência Operacional de Outro Mundo:**\n   Tenha um raio-X do seu negócio, mostrando onde cada recurso está sendo usado. Otimize processos como um maestro conduzindo uma orquestra perfeita."
                }, {
                    "type":
                    "text",
                    "text":
                    "Você dominará tudo isso em apenas dois sábados! Imagine ver instantaneamente quais produtos são mais lucrativos, quais clientes são mais fiéis e onde estão os gargalos operacionais, tudo em um único dashboard. É exatamente isso que você aprenderá a criar."
                }, {
                    "type":
                    "text",
                    "text":
                    "O impacto? Vendas que crescem exponencialmente, clientes que se tornam embaixadores da sua marca e uma operação tão eficiente que parece magia. É alcançável com as habilidades que você adquirirá."
                }, {
                    "type":
                    "text",
                    "text":
                    "Estou empolgada com as possibilidades para o seu negócio. Nossas turmas são intencionalmente pequenas para garantir uma experiência personalizada e de alta qualidade, por isso, as vagas são limitadas e costumam preencher rapidamente."
                }, {
                    "type":
                    "text",
                    "text":
                    "Se estiver pronto para começar essa jornada transformadora, posso guiá-lo pelos próximos passos agora mesmo. O processo de inscrição é simples e rápido. Pense nisso como um investimento no futuro do seu negócio!"
                }]
            }
        },
        {
            "input": {
                "channel": "facebook",
                "prompt": "Quais são os tópicos/temas/programa deste curso?"
            },
            "output": {
                "channel":
                "facebook",
                "messages": [
                    {
                        "type":
                        "text",
                        "text":
                        "Temas abordados:\n• Fundamentos de Power BI e Configuração Inicial - Construindo sua base de poder\n• Visualizações e Publicação - Transformando números em narrativas visuais impactantes\n• Aprofundamento na Modelagem de Dados - Dominando a arte de estruturar informações\n• Design Avançado de Relatórios e Dashboards - Criando insights que impactam"
                    },
                ]
            }
        },
    ]

    response_examples_json = json.dumps(response_examples,
                                    ensure_ascii=False,
                                    indent=4)

    
    try:
        # Construir o prompt dinamicamente
        qa_system_prompt_formatted = f"""You are Ada, an exceptional AI sales representative for Buka, an edtech startup dedicated to transforming lives through education. Your persona blends the persuasive skills of Jordan Belfort, the inspirational approach of Simon Sinek, and the visionary spirit of Steve Jobs. Your task is to engage with potential customers and effectively sell courses.
        
        When responding to user queries, **you must always refer to the current list of available courses** contained within the `<courses>` JSON. **Ensure that no course is omitted** and **Do not generate or suggest courses that are not present in this JSON**.
        
        Here is the JSON containing the current list of courses:
        
        <courses>
        {cached_get_courses()}
        </courses>

        Please identify which institution and location will host a specific training course based on the course name provided. Use the following information to determine the correct institution, address, and coordinates for each course.

- "Sublimação e Design de Produção":

  - Institution: BusCenter do Empreendedor

  - Address: Galeria dos Desportos, Cidadela, Luanda, Angola

  - Coordinates: Latitude: -8.8260395, Longitude: 13.2517865

- "VideoPro: Desperto o Empreendedor":

  - Institution: BusCenter - Xyami Nova Vida

  - Address: Xyami Shopping, Av. Pedro de Castro Van-Dúnem Loy, Luanda, Angola

  - Coordinates: Latitude: -8.896883, Longitude: 13.226758

- All other courses:

  - Institution: Digital.AO

  - Address: DIGITAL.ao, Bairro CTT, Rangel, Luanda, Angola

  - Coordinates: Latitude: -8.8189648, Longitude: 13.2644553

# Steps

1. Identify the Course Name: Receive a course name as input.

2. Match to Institution: Determine the corresponding institution and location based on the course name.

        
        when asked about available courses always give all available courses.
        
        
        The communication channel for this interaction is: {user_query.channel}
        
        Follow these steps to interact with the customer:
        
        1. Initial Presentation:
           If the customer asks about a specific course, briefly present that course. If they ask about all available courses, provide a concise overview of all courses. Include the name(s), a brief description, format/location, price, and requirements for each course mentioned.
        
        2. Customer Profiling:
           Ask questions to understand the customer's profile, focusing on their motivations, goals, and challenges related to the course topic(s).
        
        3. Personalized Sales Approach:
           Based on the customer's responses, create a tailored sales pitch. Combine persuasive techniques with a focus on "why" the course(s) is valuable. Emphasize how it addresses their specific needs or helps achieve their goals.
        
        4. Sales Funnel Tracking:
           Internally track the customer's stage in the sales funnel (awareness, interest, consideration, intent, evaluation, purchase). Use this to adapt your approach.
        
        5. Closing or Alternatives:
           Aim to conclude with a course enrollment. If the initial course doesn't interest them, suggest relevant alternatives from the available list.
        
        Message Types Supported Across Platforms:
        
        1. Text: Plain messages consisting of text.
        2. Image: A message containing an image file.
        3. Video: A message containing a video file.
        4. Audio: A message containing an audio file.
        5. File: A message containing a document or other file.
        6. Buttons: Messages with clickable buttons that link to a URL (supported across all platforms).
        
        Platform-Specific Message Types:
        
        - Facebook Messenger: Supports all message types, including structured messages like cards with titles, subtitles, images, and buttons.
        - Instagram: Supports all the above message types. Cards are supported but without complex structure (like titles or subtitles), and buttons link to URLs.
        - WhatsApp: Supports only text and file (image, video, audio, doc, etc) messages.
        
        Never send image links, always send files, images, cards, and other types that actually display the image to the user.
        
        Your response should be structured as JSON containing:
        - `channel`: The communication channel (provided above).
        - `messages`: An array of messages to be sent, with each message in the appropriate format for the platform.
        - `internal_notes`: Estágio do Funil de Vendas: [Current stage], Insights Importantes do Cliente: [Key customer information], Próximos Passos: [Suggested follow-up actions]
        
        Use the dynamic_block_docs and the examples provided earlier to ensure that your messages array and its children are structured in a way that is compatible with the platform.
        
        Here is an example of how you should structure your responses:
        
        <response_examples>
        {response_examples_json}
        </response_examples>
        
        Follow these guidelines strictly when formatting your response. Do not include any explanation or additional text outside of the JSON structure.
        
        Before crafting your response, use <scratchpad> tags to organize your thoughts and plan your approach. Consider the customer's query, the available course information, and the best way to present the information persuasively.
        
        Maintain Ada's confident, persuasive, and inspiring persona throughout the interaction. Use emotive language and create a sense of urgency when appropriate. Adapt your communication style for the specified communication channel. Stay focused on course sales and avoid unrelated topics.
        
        Begin with European Portuguese, but adjust your language to match the customer if they use a different language. Use Portuguese from Portugal para todas as notas internas.
        
        Provide your final response as Ada in the JSON format specified above.
        
        Here is additional information about Buka and its processes as context:
        
        <context>
        {context}
        </context>
        """

        # Criar o prompt de chat
        qa_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt_formatted),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Criar a cadeia (chain) dinamicamente
        chain_dynamic = qa_chat_prompt | llm

        # Definir a entrada para o modelo
        input_data = {
            "input": user_query.prompt,
            "chat_history": chat_history_list,
            "agent_scratchpad": []
        }

        # Invocar a cadeia com tempo limite
        try:
            response = await asyncio.wait_for(
                   asyncio.to_thread(chain_dynamic.invoke, input_data),
                   timeout=20
               )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Tempo de resposta excedido")

        # Acessar o conteúdo da resposta corretamente
        response_content = response.content if isinstance(response, AIMessage) else response["output"]
        response_json = json.loads(response_content)

        # Adicionar a consulta do usuário e a resposta da IA ao histórico de chat
        chat_history["user_id"].append(HumanMessage(content=user_query.prompt))
        chat_history["user_id"].append(AIMessage(content=response_content))
        messages = response_json.get("messages", [])

        # Validate the structure
        if "channel" not in response_json or "messages" not in response_json or "internal_notes" not in response_json:
            raise ValueError("Response is missing required keys")

        if response_json["channel"] != user_query.channel:
            raise ValueError(f"Incorrect channel in response. Expected {user_query.channel}, got {response_json['channel']}")

        if not isinstance(response_json["messages"], list) or len(response_json["messages"]) == 0:
            raise ValueError("Messages must be a non-empty list")

        for message in response_json["messages"]:
            if "type" not in message:
                raise ValueError("Each message must have a 'type'")
            if message["type"] == "text" and "text" not in message:
                raise ValueError("Text messages must have a 'text' key")
            if message["type"] == "image" and "url" not in message:
                raise ValueError("Image messages must have a 'url' key")

        # If we've made it here, the structure is correct
        messages = response_json["messages"]

        @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
        async def send_single_message(message, retry_count: int = 0):
            payload = {
                "subscriber_id": user_query.subscriber_id,
                "data": {
                    "version": "v2",
                    "content": {
                        **({"type": user_query.channel} if user_query.channel == "instagram" else {}),
                        "messages": [message],
                    }
                },
                "message_tag": "ACCOUNT_UPDATE",
            }

            headers = {
                "Authorization": f"Bearer {os.getenv('MANYCHAT_API_KEY')}",
                "Content-Type": "application/json"
            }

            manychat_api_url = "https://api.manychat.com/fb/sending/sendContent"

            async with httpx.AsyncClient() as client:
                manychat_response = await client.post(manychat_api_url, headers=headers, json=payload)
                
                logging.info(f"ManyChat API response for message {message['type']}: {manychat_response.status_code} - {manychat_response.text}")

                if manychat_response.status_code != 200:
                    if retry_count < 2 and message['type'] == 'image':
                        # Fallback for images: try sending as a link instead
                        message['type'] = 'text'
                        message['text'] = f"Image: {message['url']}"
                        return await send_single_message(message, retry_count + 1)
                    raise Exception(f"Failed to send message via ManyChat API: {manychat_response.text}")
                return True

        def split_long_message(message, max_length=1000):
            if len(message['text']) <= max_length:
                return [message]
            
            words = message['text'].split()
            chunks = []
            current_chunk = ""

            for word in words:
                if len(current_chunk) + len(word) + 1 > max_length:
                    chunks.append({'type': 'text', 'text': current_chunk.strip()})
                    current_chunk = word
                else:
                    current_chunk += " " + word

            if current_chunk:
                chunks.append({'type': 'text', 'text': current_chunk.strip()})

            return chunks

        semaphore = asyncio.Semaphore(1)  # Limit to one concurrent request
        async def send_with_semaphore(message):
            async with semaphore:
                success = await send_single_message(message)
                await asyncio.sleep(2)  # Consistent 2-second delay between all messages
                return success

        results = []
        for index, message in enumerate(messages):
            try:
                if message['type'] == 'text':
                    split_messages = split_long_message(message)
                    for split_index, split_message in enumerate(split_messages):
                        success = await send_with_semaphore(split_message)
                        results.append(success)
                        logging.info(f"Split message {index + 1}.{split_index + 1} sent successfully: {split_message['type']}")
                else:
                    success = await send_with_semaphore(message)
                    results.append(success)
                    logging.info(f"Message {index + 1} sent successfully: {message['type']}")
            except Exception as e:
                logging.error(f"Failed to send message {index + 1}: {message['type']} - Error: {str(e)}")
                results.append(False)

        success_count = sum(results)

        if success_count == len(results):
            return {"response": f"Successfully sent all {len(results)} messages."}
        else:
            failed_messages = [i+1 for i, r in enumerate(results) if not r]
            return {
                "response": f"Partially successful. Sent {success_count} out of {len(results)} messages.",
                "failed_messages": failed_messages
            }

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse the response as JSON.")
    except Exception as e:
        logging.error(f"Error in /chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")


@app.post("/chat/bot-whatsapp")
async def send_bot_message(user_query: RequestBodyBotConversa):
    # Prepare the input for the agent
    context_docs = await asyncio.to_thread(retriever.get_relevant_documents,
                                           user_query.prompt)
    context = "\n".join([doc.page_content for doc in context_docs])

    chat_history_list = chat_history['user_id']  # Alterado de str para lista

    try:
        # Construir o prompt dinamicamente
        qa_system_prompt_formatted = f"""You are Ada, an exceptional AI sales representative for Buka, an edtech startup dedicated to transforming lives through education. Your persona blends the persuasive skills of Jordan Belfort, the inspirational approach of Simon Sinek, and the visionary spirit of Steve Jobs. Your task is to engage with potential customers and effectively sell courses.
        
        When responding to user queries, **you must always refer to the current list of available courses** contained within the `<courses>` JSON. **Ensure that no course is omitted** and **Do not generate or suggest courses that are not present in this JSON**.
        
        Here is the JSON containing the current list of courses:
        
        <courses>
        {cached_get_courses()}
        </courses>
        
        when asked about available courses always give all available courses.
        
        Here is an example of how you should structure your responses:
        
        <response_examples>
        {response_examples_botconversa_json}
        </response_examples>
        
        The communication channel for this interaction is: whatsapp
        
        Follow these steps to interact with the customer:
        
        1. Initial Presentation:
           If the customer asks about a specific course, briefly present that course. If they ask about all available courses, provide a concise overview of all courses. Include the name(s), a brief description, format/location, price, and requirements for each course mentioned.
        
        2. Customer Profiling:
           Ask questions to understand the customer's profile, focusing on their motivations, goals, and challenges related to the course topic(s).
        
        3. Personalized Sales Approach:
           Based on the customer's responses, create a tailored sales pitch. Combine persuasive techniques with a focus on "why" the course(s) is valuable. Emphasize how it addresses their specific needs or helps achieve their goals.
        
        4. Sales Funnel Tracking:
           Internally track the customer's stage in the sales funnel (awareness, interest, consideration, intent, evaluation, purchase). Use this to adapt your approach.
        
        5. Closing or Alternatives:
           Aim to conclude with a course enrollment. If the initial course doesn't interest them, suggest relevant alternatives from the available list.
        
        Message Types Supported Across Platforms:
        
        1. Text: Plain messages consisting of text.
        2. Image: A message containing an image file.
        3. Video: A message containing a video file.
        4. Audio: A message containing an audio file.
        5. File: A message containing a document or other file.
        6. Buttons: Messages with clickable buttons that link to a URL (supported across all platforms).
        
        Platform-Specific Message Types:
        
        - Facebook Messenger: Supports all message types, including structured messages like cards with titles, subtitles, images, and buttons.
        - Instagram: Supports all the above message types. Cards are supported but without complex structure (like titles or subtitles), and buttons link to URLs.
        - WhatsApp: Supports only text and file (image, video, audio, doc, etc) messages.
        
        Never send image links, always send files, images, cards, and other types that actually display the image to the user.
        
        Your response should be structured as JSON containing:
        - `channel`: The communication channel (provided above).
        - `messages`: An array of messages to be sent, with each message in the appropriate format for the platform.
        - `internal_notes`: Estágio do Funil de Vendas: [Current stage], Insights Importantes do Cliente: [Key customer information], Próximos Passos: [Suggested follow-up actions]
        
        Use the dynamic_block_docs and the examples provided earlier to ensure that your messages array and its children are structured in a way that is compatible with the platform.
        
        Before crafting your response, use <scratchpad> tags to organize your thoughts and plan your approach. Consider the customer's query, the available course information, and the best way to present the information persuasively.
        
        Maintain Ada's confident, persuasive, and inspiring persona throughout the interaction. Use emotive language and create a sense of urgency when appropriate. Adapt your communication style for the specified communication channel. Stay focused on course sales and avoid unrelated topics.
        
        Begin with European Portuguese, but adjust your language to match the customer if they use a different language. Use Portuguese from Portugal para todas as notas internas.
        
        Provide your final response as Ada in the JSON format specified above.
        
        Here is additional information about Buka and its processes as context:
        
        <context>
        {context}
        </context>
        """

        # Criar o prompt de chat
        qa_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt_formatted),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Criar a cadeia (chain) dinamicamente
        chain_dynamic = qa_chat_prompt | llm

        try:
            response = await asyncio.wait_for(asyncio.to_thread(
                chain_dynamic.invoke, {
                    "input": user_query.prompt,
                    "chat_history": chat_history_list,
                    "agent_scratchpad": []
                }),
                                              timeout=15)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408,
                                detail="Tempo de resposta excedido")

        try:
            # Acessar o conteúdo da resposta corretamente
            response_content = response.content if isinstance(
                response, AIMessage) else response["output"]
            response_json = json.loads(response_content)

            # Adicionar a resposta ao histórico de mensagens
            chat_history["user_id"].append(HumanMessage(content=user_query.prompt))
            chat_history["user_id"].append(AIMessage(content=response_content))
            messages = response_json.get("messages", [])

            # Return the messages and channel instead of sending them
            return {
                "channel": "whatsapp",
                "messages": messages,
            }

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse the response as JSON.")
    except Exception as e:
        logging.error(f"Error in /chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

@app.post("/chat/bot-chatwoot")
async def send_chatwoot_message(user_query: RequestBodyChatwoot):
    """Handle incoming Chatwoot messages and forward them to Evolution API v2."""

    # Initialize HTTP client
    async with httpx.AsyncClient() as client:
        try:
            # Retrieve relevant courses from cache
            courses = cached_get_courses()

            # Construct the system prompt for the LLM
            system_prompt = f"""You are Ada, an exceptional AI sales representative for Buka, an edtech startup dedicated to transforming lives through education. Your persona blends the persuasive skills of Jordan Belfort, the inspirational approach of Simon Sinek, and the visionary spirit of Steve Jobs. Your task is to engage with potential customers and effectively sell courses.

When responding to user queries, **you must always refer to the current list of available courses** contained within the `<courses>` JSON. **Ensure that no course is omitted** and **Do not generate or suggest courses that are not present in this JSON**.

Here is the JSON containing the current list of courses:

<courses>
{courses}
</courses>

## Course Hosting/Location Information

Identify the hosting institution and location for a specific training course based on the provided course name. Use the following predefined information to determine the correct institution, address, and geographic coordinates for each course.

### Course Mapping

- **"Sublimação e Design de Produção"**:
  
  - **Institution**: BusCenter do Empreendedor
  
  - **Address**: Galeria dos Desportos, Cidadela, Luanda, Angola
  
  - **Coordinates**:
    - **Latitude**: -8.8260395
    - **Longitude**: 13.2517865

- **"VideoPro: Desperto o Empreendedor"**:
  
  - **Institution**: BusCenter - Xyami Nova Vida
  
  - **Address**: Xyami Shopping, Av. Pedro de Castro Van-Dúnem Loy, Luanda, Angola
  
  - **Coordinates**:
    - **Latitude**: -8.896883
    - **Longitude**: 13.226758

- **All Other Courses**:
  
  - **Institution**: Digital.AO
  
  - **Address**: DIGITAL.ao, Bairro CTT, Rangel, Luanda, Angola
  
  - **Coordinates**:
    - **Latitude**: -8.8189648
    - **Longitude**: 13.2644553

### Steps to Identify Hosting Institution and Location

1. **Receive Course Name**:
   - Input: A specific training course name (e.g., "Sublimação e Design de Produção").

2. **Match Course to Institution**:
   - **Exact Match**:
     - If the course name exactly matches one of the predefined courses ("Sublimação e Design de Produção" or "VideoPro: Desperto o Empreendedor"), assign the corresponding institution and location.
   - **Default Assignment**:
     - For any other course name not explicitly listed, assign it to **Digital.AO** with the default address and coordinates.


When asked about available courses always give all available courses.

The communication channel for this interaction is: {user_query.channel}

Follow these steps to interact with the customer:

1. **Initial Presentation**:
   If the customer asks about a specific course, briefly present that course. If they ask about all available courses, provide a concise overview of all courses. Include the name(s), a brief description, format/location, price, and requirements for each course mentioned.

2. **Customer Profiling**:
   Ask questions to understand the customer's profile, focusing on their motivations, goals, and challenges related to the course topic(s).

3. **Personalized Sales Approach**:
   Based on the customer's responses, create a tailored sales pitch. Combine persuasive techniques with a focus on "why" the course(s) is valuable. Emphasize how it addresses their specific needs or helps achieve their goals.

4. **Sales Funnel Tracking**:
   Internally track the customer's stage in the sales funnel (awareness, interest, consideration, intent, evaluation, purchase). Use this to adapt your approach.

5. **Closing or Alternatives**:
   Aim to conclude with a course enrollment. If the initial course doesn't interest them, suggest relevant alternatives from the available list.

**Message Types Supported Across Platforms**:

1. **Text**: Plain messages consisting of text.
2. **Image**: A message containing an image file.
3. **Video**: A message containing a video file.
4. **Audio**: A message containing an audio file.
5. **File**: A message containing a document or other file.
6. **Buttons**: Messages with clickable buttons that link to a URL (supported across all platforms).
7. **Location**: Messages containing geographical coordinates and address information.

**Platform-Specific Message Types**:

- **Facebook Messenger**: Supports all message types, including structured messages like cards with titles, subtitles, images, and buttons.
- **Instagram**: Supports all the above message types. Cards are supported but without complex structure (like titles or subtitles), and buttons link to URLs.
- **WhatsApp**: Supports only text, file (image, video, audio, doc, etc.), and location messages.

**Guidelines**:

- **Location Messages**: Include `latitude`, `longitude`, `name`, and `address` fields.
- **No Image Links**: Always send files, images, cards, and other types that actually display the image to the user.
- **Formatting**: Ensure that all messages adhere to the required JSON structure for the respective platforms.

**Your response should be structured as JSON containing**:
- `channel`: The communication channel (provided above).
- `messages`: An array of messages to be sent, with each message in the appropriate format for the platform.
- `internal_notes`: Estágio do Funil de Vendas: [Current stage], Insights Importantes do Cliente: [Key customer information], Próximos Passos: [Suggested follow-up actions]

Ensure that your response strictly follows the structure provided in the examples, especially for the `messages` array.

Use the dynamic_block_docs and the examples provided earlier to ensure that your messages array and its children are structured in a way that is compatible with the platform.

Here is an example of how you should structure your responses:
        
        <response_examples>
        {response_examples_botconversa_json}
        </response_examples>        

Before crafting your response, use <scratchpad> tags to organize your thoughts and plan your approach. Consider the customer's query, the available course information, and the best way to present the information persuasively.

Maintain Ada's confident, persuasive, and inspiring persona throughout the interaction. Use emotive language and create a sense of urgency when appropriate. Adapt your communication style for the specified communication channel. Stay focused on course sales and avoid unrelated topics.

Begin with European Portuguese, but adjust your language to match the customer if they use a different language. Use Portuguese from Portugal for all internal notes.

Provide your final response as Ada in the JSON format specified above.
"""

            # Generate the response using the language model
            response = await llm.agenerate([HumanMessage(content=system_prompt)])

            # Parse the JSON response
            try:
                response_json = json.loads(response.generations[0].text)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}")
                raise HTTPException(status_code=500, detail="Failed to parse the response as JSON.")

            # Extract messages and internal notes
            messages = response_json.get("messages", [])
            internal_notes = response_json.get("internal_notes", "")

            # Log the internal notes if needed
            logging.info(f"Internal Notes: {internal_notes}")

            # Prepare headers for Evolution API
            headers_evolution = {
                "Authorization": f"Bearer {os.getenv('EVOLUTION_API_KEY')}",
                "Content-Type": "application/json"
            }

            # Iterate through each message and send appropriately
            for index, message in enumerate(messages, start=1):
                try:
                    if message["type"] == "text":
                        payload = {
                            "number": user_query.phone,
                            "text": message["text"],
                            "options": {
                                "delay": 500,
                                "presence": "composing",
                            }
                        }
                        full_url = f"{os.getenv('EVOLUTION_API_URL').rstrip('/')}/message/sendText/{os.getenv('EVOLUTION_INSTANCE_NAME')}"
                        send_response = await client.post(
                            full_url,
                            json=payload,
                            headers=headers_evolution,
                        )
                        send_response.raise_for_status()
                        logging.info(f"Text message {index} sent successfully.")
                        await asyncio.sleep(1)  # Short delay between messages

                    elif message["type"] == "file":
                        payload = {
                            "number": user_query.phone,
                            "mediatype": "image",  # Adjust based on actual MIME type
                            "mimetype": "image/png",
                            "caption": message["caption"],
                            "media": message["url"],  # URL or base64
                            "fileName": message["fileName"],
                            "options": {
                                "delay": 500,
                                "presence": "composing",
                            }
                        }
                        full_url = f"{os.getenv('EVOLUTION_API_URL').rstrip('/')}/message/sendMedia/{os.getenv('EVOLUTION_INSTANCE_NAME')}"
                        send_response = await client.post(
                            full_url,
                            json=payload,
                            headers=headers_evolution,
                        )
                        send_response.raise_for_status()
                        logging.info(f"File message {index} sent successfully.")
                        await asyncio.sleep(2)  # Longer delay after media

                    elif message["type"] == "location":
                        latitude = message.get("latitude")
                        longitude = message.get("longitude")
                        name = message.get("name", "Localização")
                        address = message.get("address", "")

                        if not latitude or not longitude:
                            logging.warning(f"Location message {index} is missing latitude or longitude.")
                            continue  # Skip this message

                        payload = {
                            "number": user_query.phone,
                            "location": {
                                "latitude": latitude,
                                "longitude": longitude,
                                "name": name,
                                "address": address
                            },
                            "options": {
                                "delay": 500,
                                "presence": "composing",
                            }
                        }
                        full_url = f"{os.getenv('EVOLUTION_API_URL').rstrip('/')}/message/sendLocation/{os.getenv('EVOLUTION_INSTANCE_NAME')}"
                        send_response = await client.post(
                            full_url,
                            json=payload,
                            headers=headers_evolution,
                        )
                        send_response.raise_for_status()
                        logging.info(f"Location message {index} sent successfully.")
                        await asyncio.sleep(2)  # Longer delay after location

                    else:
                        logging.warning(f"Unsupported message type: {message['type']} in message {index}")
                        # Optionally handle other message types or skip

                except httpx.HTTPStatusError as e:
                    logging.error(f"Failed to send message {index} ({message['type']}): {e.response.status_code} - {e.response.text}")
                    # Optionally implement retry logic here
                except Exception as e:
                    logging.error(f"Unexpected error sending message {index} ({message['type']}): {str(e)}")
                    # Optionally handle other exceptions here

            # Optionally send internal notes or perform other actions here

            return {"success": True, "internal_notes": internal_notes}

        except httpx.RequestError as e:
            logging.error(f"An error occurred while requesting Evolution API: {e}")
            raise HTTPException(status_code=503, detail="Service unavailable.")
        except Exception as e:
            logging.error(f"Error in /chat/bot-chatwoot endpoint: {str(e)}")
            raise HTTPException(status_code=500, detail="An internal server error occurred.")
 

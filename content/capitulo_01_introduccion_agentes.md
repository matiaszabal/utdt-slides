# Capítulo 1 — Introducción a los Agentes IA
## Arquitectura de Agentes IA · UTDT 2026

---

## Agenda

- Definición de AI Agents
- La revolución del Pretraining
- Tipos de agentes
- Model Selection
- De sincrónico a asincrónico
- Aplicaciones prácticas y casos de uso
- Workflows vs Agentes
- Principios para sistemas agénticos efectivos
- Organización estratégica
- Agentic Frameworks: panorama y elección

---

## ¿Qué es un AI Agent?

Un **AI Agent** es un sistema que percibe su entorno, razona sobre él y ejecuta acciones para alcanzar un objetivo.

Tres propiedades esenciales:

- **Percepción** — recibe inputs (texto, datos, eventos)
- **Razonamiento** — decide qué hacer usando un LLM como motor
- **Acción** — ejecuta herramientas, llama APIs, modifica estado

> "An agent is any system that uses an LLM to decide the control flow of an application."
> — Building Applications with AI Agents, O'Reilly

---

## De Modelos a Agentes

```
Input → Modelo → Output
```

vs.

```
Input → Agente → [Razonamiento → Herramientas → Observación] × N → Output
```

La diferencia clave: el agente **itera**. No produce una respuesta en un solo paso; construye su respuesta a través de un ciclo de razonamiento y acción.

Esto habilita tareas que antes requerían código procedural complejo.

---

## La Revolución del Pretraining

Los LLMs modernos son posibles gracias a tres factores convergentes:

| Factor | Descripción |
|---|---|
| **Escala de datos** | Billones de tokens de texto web, código, libros |
| **Arquitectura Transformer** | Atención paralela, captura dependencias largas |
| **Compute masivo** | GPUs/TPUs en clusters de miles de unidades |

El **pretraining** produce modelos con conocimiento general del mundo.  
El **fine-tuning** + **RLHF** los convierte en asistentes útiles y seguros.

---

## Pretraining: Lo que cambia para los Agentes

Antes del pretraining a escala:

- Cada tarea requería un modelo entrenado específicamente
- Integrar razonamiento + herramientas era inviable

Después:

- Un modelo puede **razonar sobre herramientas** sin haberlas visto antes
- Puede **seguir instrucciones complejas** de orquestación
- Puede **autocorregirse** al observar resultados de sus acciones

El pretraining es la base que hace posible el paradigma agéntico.

---

## Tipos de Agentes: Taxonomía

| Tipo | Descripción | Cuándo usarlo |
|---|---|---|
| **Reflex** | Mapea input → acción directa | Reglas simples, latencia mínima |
| **ReAct** | Reason + Act en ciclos | Tareas con herramientas, debugging |
| **Planner-Executor** | Planifica primero, ejecuta después | Tareas largas y complejas |
| **Query Decomposition** | Divide preguntas complejas | RAG multi-step, análisis |
| **Reflection** | Critica y refina su propio output | Alta calidad requerida |
| **Deep Research** | Multi-hop, multi-fuente | Investigación exhaustiva |

---

## Agente Reflex

El más simple: input → output sin estado interno.

```python
from google.adk.agents import LlmAgent

reflex_agent = LlmAgent(
    name="classifier",
    model="gemini-2.0-flash",
    instruction="""Classify the customer message as: complaint, query, or praise.
    Respond with only one word.""",
)
```

**Ventajas:** latencia mínima, predecible, barato  
**Limitaciones:** no maneja ambigüedad, sin memoria, sin herramientas

---

## Agente ReAct (Reason + Act)

El patrón más usado en producción. Ciclo:

```
1. Recibe tarea
2. Razona: ¿qué necesito hacer?
3. Llama herramienta
4. Observa resultado
5. Razona de nuevo → ¿terminé?
6. Si no: volver a 2
7. Respuesta final
```

El LLM actúa como motor de decisión en cada paso.  
ADK2 implementa este ciclo automáticamente con `LlmAgent` + `tools`.

---

## Agente ReAct: ejemplo en ADK2

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

def search_web(query: str) -> str:
    """Search the web for current information."""
    # integración real con SerpAPI, Tavily, etc.
    return f"Results for: {query}"

agent = LlmAgent(
    name="research_agent",
    model="gemini-2.0-flash",
    instruction="Answer questions using web search when needed.",
    tools=[search_web],
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="demo", session_service=session_service)
```

---

## Planner-Executor

Separa **qué hacer** de **cómo hacerlo**.

```
Usuario → Planner (LLM) → Plan estructurado
                              ↓
                    Executor (LLM o código)
                    → Paso 1 → Resultado 1
                    → Paso 2 → Resultado 2
                    → ...
                    → Respuesta final
```

**Ventaja:** el plan es auditable antes de ejecutar  
**Desventaja:** latencia adicional, el plan puede ser subóptimo

En ADK2: implementado con `SequentialAgent` o con un agente de planning que genera un plan para otro agente.

---

## Query Decomposition

Especializado para responder preguntas complejas sobre múltiples fuentes.

```
"¿Cómo impactó la pandemia en los ingresos de FAANG entre 2020 y 2023?"
                              ↓
        Descompone en sub-queries:
        1. Ingresos Apple 2020-2023
        2. Ingresos Google 2020-2023
        3. Ingresos Meta 2020-2023
        4. Contexto pandemia tech industry
                              ↓
        Ejecuta en paralelo → Consolida → Respuesta
```

En ADK2: `ParallelAgent` para los sub-queries + `LlmAgent` para consolidar.

---

## Agente de Reflexión

Agrega un loop de auto-crítica:

```
Draft → Critic (mismo o distinto LLM) → ¿Suficientemente bueno?
                                              ↓ No
                                        Refine Draft
                                              ↓ Sí
                                        Output final
```

Útil para: código generado, análisis legales, redacción técnica.

En ADK2: `LoopAgent` con condición de salida basada en score de calidad.

```python
from google.adk.agents import LoopAgent, LlmAgent

writer = LlmAgent(name="writer", model="gemini-2.0-flash",
                  instruction="Write a technical explanation.")
critic = LlmAgent(name="critic", model="gemini-2.0-flash",
                  instruction="Score the draft 1-10. If >= 8, say DONE.")

reflection_agent = LoopAgent(
    name="reflection",
    sub_agents=[writer, critic],
    max_iterations=5,
)
```

---

## Deep Research Agent

El tipo más complejo: combina todos los patrones anteriores.

Características:
- Multi-hop: cada resultado dispara nuevas búsquedas
- Multi-fuente: web, bases de datos, documentos internos
- Síntesis: consolida decenas de fuentes en un reporte coherente
- Verificación: cruza información entre fuentes

**Latencia:** segundos a minutos  
**Costo:** alto (muchas llamadas al LLM)  
**Valor:** reemplaza horas de trabajo de analista

---

## Model Selection: Los Criterios que Importan

No existe el modelo perfecto. La elección depende del caso de uso:

| Criterio | Modelos rápidos/baratos | Modelos potentes |
|---|---|---|
| **Latencia** | < 1 segundo | 5-30 segundos |
| **Costo** | $0.001-0.01 / 1K tokens | $0.01-0.15 / 1K tokens |
| **Razonamiento** | Tareas simples | Tareas complejas |
| **Context window** | 8K-32K | 128K-1M+ |
| **Uso ideal** | Reflex, clasificación | Planner, Reflection |

En este curso: `gemini-2.0-flash` (equilibrio velocidad/capacidad, gratis con AI Studio)

---

## Model Selection: Estrategia Multi-Modelo

Un sistema agéntico puede usar **modelos distintos por rol**:

```python
# Planner: necesita razonar bien
planner = LlmAgent(
    name="planner",
    model="gemini-2.0-flash",  # o gemini-1.5-pro para casos complejos
    instruction="Create a step-by-step plan for the task."
)

# Executor: opera rápido y barato
executor = LlmAgent(
    name="executor",
    model="gemini-2.0-flash",
    instruction="Execute the given step and return results."
)
```

ADK2 soporta Gemini, OpenAI y Anthropic vía LiteLLM en el mismo sistema.

---

## De Sincrónico a Asincrónico

La mayoría de los LLMs se diseñaron para interacción sincrónica (request/response).

Los agentes introducen **asincronía necesaria**:

| Patrón | Sincrónico | Asincrónico |
|---|---|---|
| Chat | ✓ | — |
| Tool call única | ✓ | — |
| Pipeline multi-step | — | ✓ |
| Tarea de minutos | — | ✓ |
| Notificaciones push | — | ✓ |
| Human-in-the-loop | — | ✓ |

---

## Asincronía en ADK2

ADK2 soporta tanto ejecución sincrónica como asincrónica:

```python
# Sincrónico (simple, para demos y pipelines cortos)
for event in runner.run(
    user_id="u1",
    session_id=session.id,
    new_message=content
):
    if event.is_final_response() and event.content:
        print(event.content.parts[0].text)

# Asincrónico (para producción, UI reactiva)
async for event in runner.run_async(
    user_id="u1",
    session_id=session.id,
    new_message=content
):
    if event.is_final_response() and event.content:
        print(event.content.parts[0].text)
```

---

## Aplicaciones Prácticas

Casos de uso que veremos en el curso:

| Dominio | Aplicación | Tipo de Agente |
|---|---|---|
| **Customer Support** | Resolución autónoma de tickets | ReAct |
| **Data Analysis** | Análisis de datasets con código | Planner-Executor |
| **Content Creation** | Redacción + revisión iterativa | Reflection |
| **Research** | Síntesis de literatura técnica | Deep Research |
| **DevOps** | Diagnóstico y remediación | ReAct multi-tool |
| **Finance** | Due diligence automatizada | Query Decomposition |

---

## ¿Cuándo NO usar un Agente?

Importante: los agentes no son siempre la respuesta correcta.

**Usar agente cuando:**
- La tarea requiere múltiples pasos con decisiones en el camino
- El flujo exacto no se puede predeterminar
- Se necesita adaptación a resultados intermedios

**No usar agente cuando:**
- La tarea es un único LLM call con prompt engineering
- El flujo es 100% predecible (usar pipeline fijo)
- La latencia es crítica y el costo importa mucho
- No hay tolerancia a errores no deterministas

---

## Workflows vs Agentes

Una distinción crítica para el diseño de sistemas:

| Dimensión | Workflow | Agente |
|---|---|---|
| **Control de flujo** | Determinístico, predefinido | Dinámico, decidido por el LLM |
| **Adaptabilidad** | Baja | Alta |
| **Predictibilidad** | Alta | Media |
| **Debugging** | Sencillo | Complejo |
| **Cuándo usarlo** | Procesos estables y conocidos | Tareas abiertas y variables |

La mayoría de los sistemas de producción combinan ambos: **workflows con agentes embebidos**.

---

## Principios para Sistemas Agénticos Efectivos

Los 5 principios que guían el diseño en este curso:

1. **Minimal footprint** — hacer lo mínimo necesario para cumplir la tarea
2. **Human-in-the-loop** — saber cuándo pedir confirmación
3. **Observabilidad** — si no lo podés ver, no lo podés mejorar
4. **Fail gracefully** — el error es esperado, el crash no
5. **Iterative design** — empezar simple, medir, complejizar con evidencia

---

## Principio 1: Minimal Footprint

Los agentes con acceso excesivo a herramientas son un riesgo:

```python
# MAL: el agente puede hacer cualquier cosa
agent = LlmAgent(
    tools=[read_db, write_db, delete_records, send_email, deploy_code]
)

# BIEN: el agente tiene solo lo que necesita
read_agent = LlmAgent(
    tools=[read_db, search_docs],
    instruction="Only retrieve and analyze data. Never modify anything."
)
```

Principio del mínimo privilegio aplicado a agentes IA.

---

## Principio 2: Observabilidad desde el Día 1

ADK2 incluye OpenTelemetry built-in. En el curso usamos:

```
Agente ADK2
    → OpenTelemetry traces
        → Tempo (storage)
        → Grafana (visualización)
    → Logs
        → Loki (storage)
        → Grafana (visualización)
```

Stack completo con Docker Compose. Veremos trazas de cada tool call, latencias por paso, y costos de tokens.

---

## Organización Estratégica para el Éxito

Adoptar sistemas agénticos requiere cambios organizacionales:

| Área | Cambio necesario |
|---|---|
| **Equipos** | Roles nuevos: Agent Engineer, Prompt Engineer |
| **Procesos** | Evaluación continua de outputs del agente |
| **Gobernanza** | Políticas de herramientas permitidas por agente |
| **Cultura** | Tolerancia al no-determinismo controlado |
| **Métricas** | KPIs de calidad de output, no solo uptime |

El mayor bloqueador no es técnico: es la resistencia a ceder control a un sistema autónomo.

---

## Agentic Frameworks: El Panorama

| Framework | Origen | Fortaleza | Limitación |
|---|---|---|---|
| **LangGraph** | LangChain | Grafos de estado explícitos | Verboso, steep learning curve |
| **AutoGen** | Microsoft | Multi-agent conversations | Menos control fino |
| **CrewAI** | Comunidad | API simple, roles claros | Menos flexible para casos complejos |
| **ADK2** | Google | Integración Gemini, producción-ready | Ecosistema más nuevo |

---

## Comparativa: Frameworks vs ADK2

| Concepto | LangGraph / AutoGen / CrewAI | ADK2 |
|---|---|---|
| Grafo de estados | `StateGraph` + nodos explícitos | Workflow Runtime con grafos implícitos |
| Definición de tools | `@tool` decorator (LangChain) | Función Python plain con docstring |
| Memoria | `MemorySaver`, `ConversationBuffer` | `InMemorySessionService` / `InMemoryMemoryService` |
| Multi-agente | `AgentExecutor` anidado | `SequentialAgent`, `ParallelAgent`, `LoopAgent` |
| Observabilidad | LangSmith (pago) | OpenTelemetry built-in (gratis) |
| LLM soportados | Muchos vía LangChain | Gemini nativo + OpenAI/Anthropic vía LiteLLM |
| Curva de aprendizaje | Alta | Media |

---

## ¿Por qué ADK2 en este curso?

- **Producción-ready:** diseñado por Google para sus propios sistemas agénticos
- **Observabilidad nativa:** OpenTelemetry sin configuración extra
- **API limpia:** menos boilerplate que LangGraph para casos comunes
- **Integración Gemini:** acceso directo a modelos de última generación
- **Gratis para aprender:** Google AI Studio API key sin costo para volúmenes moderados
- **Multi-LLM:** no te ata a un solo proveedor

---

## Setup del Curso

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate

# Instalar ADK2
pip install google-adk==2.2.0

# Configurar API key
export GOOGLE_API_KEY="tu_api_key_de_ai_studio"

# Verificar instalación
python -c "from google.adk.agents import LlmAgent; print('OK')"
```

API key gratuita: https://aistudio.google.com/apikey

---

## Tu Primer Agente en ADK2

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

agent = LlmAgent(
    name="my_first_agent",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant for UTDT students.",
)

async def main():
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="utdt", session_service=session_service)
    session = await session_service.create_session(app_name="utdt", user_id="u1")
    content = types.Content(role="user",
                            parts=[types.Part(text="¿Qué es un agente IA?")])
    for event in runner.run(user_id="u1", session_id=session.id, new_message=content):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

---

## Key Takeaways — Capítulo 1

- Un **AI Agent** percibe, razona y actúa en ciclos iterativos
- El **pretraining** a escala es lo que hace posible el paradigma agéntico
- Existen 6 tipos principales: Reflex, ReAct, Planner-Executor, Query Decomposition, Reflection, Deep Research
- **No todo necesita un agente**: workflows determinísticos son más predecibles y baratos
- La **observabilidad** y el **minimal footprint** son principios no negociables
- Usamos **ADK2 con `gemini-2.0-flash`**: API limpia, observabilidad nativa, gratis para aprender
- El mayor desafío de adopción es organizacional, no técnico

---

## Próxima Clase

**Capítulo 2 — Diseño de Sistemas de Agentes**

- Arquitectura de un agente completo con ADK2
- Componentes core: modelo, tools, memoria, orquestación
- Trade-offs de diseño: velocidad, costo, confiabilidad
- Patrones: single-agent vs multi-agent
- Buenas prácticas de evaluación

> Traer instalado: Python 3.11+, google-adk==2.2.0, Google AI Studio API key

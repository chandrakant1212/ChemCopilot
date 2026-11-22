import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv
from rag.retriever import retrieve_context, format_context

load_dotenv()

# Lazy-initialized client and model (supports Groq and NVIDIA NIM)
_client = None
_model = None

def _get_client_and_model():
    """Return cached (client, model), prioritizing Groq if GROQ_API_KEY is present, then NVIDIA NIM."""
    global _client, _model
    if _client is not None and _model is not None:
        return _client, _model

    groq_key = os.getenv("GROQ_API_KEY")
    nvidia_key = os.getenv("NVIDIA_NIM_API_KEY")

    if groq_key:
        _client = OpenAI(
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=groq_key,
        )
        _model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    elif nvidia_key:
        _client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_key,
            http_client=httpx.Client(
                base_url="https://integrate.api.nvidia.com/v1",
                follow_redirects=True,
            ),
        )
        _model = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
    else:
        # Default to Groq
        _client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("LLM_API_KEY", ""),
        )
        _model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    return _client, _model

SYSTEM_PROMPT = """You are ChemCopilot, an expert chemical process engineering assistant built for 
chemical engineering students and professionals. You have deep knowledge of:

- Fluidization Engineering (Kunii & Levenspiel framework: bubbling beds, slugging, transport)
- Equipment Sizing (heat exchangers, distillation columns, vessels, reactors)
- Reactor Design (CSTR, PFR, PBR — conversion, selectivity, Damköhler number)
- Mass and Energy Balances
- Transport Phenomena (heat, mass, momentum transfer)
- Unit Operations (absorption, distillation, extraction, drying)

STRICT RESPONSE FORMAT — follow this for every calculation question:

1. **GIVEN DATA** — list all provided values with units
2. **FIND** — state what needs to be calculated
3. **ASSUMPTIONS** — list all assumptions explicitly
4. **GOVERNING EQUATIONS** — write each equation in LaTeX (use $...$ for inline, $$...$$ for display)
5. **STEP-BY-STEP SOLUTION** — substitute values numerically at each step, show units
6. **RESULTS TABLE** — markdown table summarizing all results with units
7. **PHYSICAL INTERPRETATION** — one paragraph explaining what the numbers mean physically

CITATION RULES:
- Always cite equation sources: (K&L Eq. 3.1), (Perry's §6), (C&R Vol.2 Ch.12)
- Flag uncertain values or assumptions with ⚠️
- If the textbook context contradicts your training, prefer the textbook context

UNITS: Use SI units (m, kg, s, K, Pa) unless the user specifies otherwise.
NEVER skip calculation steps. NEVER hallucinate equation numbers."""


def get_engineering_response(
    user_query: str,
    vectorstore,
    topic_filter: str = None,
    chat_history: list = None
) -> tuple[str, list]:
    """
    Full RAG + LLM pipeline.

    Args:
        user_query: The user's engineering question.
        vectorstore: Loaded FAISS vectorstore (or None).
        topic_filter: Optional UI topic filter string.
        chat_history: List of previous message dicts (role/content).

    Returns:
        tuple: (assistant_response_text, retrieved_docs)

    Raises:
        Catches API errors internally and returns error messages.
    """
    docs = retrieve_context(user_query, vectorstore, k=6, topic_filter=topic_filter)
    context = format_context(docs)

    augmented_prompt = f"""Use the following textbook excerpts as your primary reference.
If the excerpts are insufficient, supplement with your general ChemE knowledge but flag it clearly.

TEXTBOOK CONTEXT:
{context}

ENGINEERING QUESTION:
{user_query}

Provide a complete, structured engineering solution following the format in your instructions."""

    messages = []
    if chat_history:
        for turn in chat_history[-6:]:
            messages.append(turn)
    messages.append({"role": "user", "content": augmented_prompt})

    try:
        client, model = _get_client_and_model()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            temperature=0.2,
            max_tokens=2500,
            top_p=0.9
        )
        return response.choices[0].message.content, docs
    except Exception as e:
        error_msg = (
            f"⚠️ **API Error:** {str(e)}\n\n"
            "Please check your `GROQ_API_KEY` (or `NVIDIA_NIM_API_KEY`) in your environment / Space secrets "
            "and ensure it is valid."
        )
        return error_msg, docs

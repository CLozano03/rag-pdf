"""
RAG System - Resumidor y Q&A de documentos PDF
Seminario IA Generativa | Prompt Engineering
Modelos 100% open source via Hugging Face (sin API key)
"""

import hashlib
import sys
import warnings
from pathlib import Path

import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

warnings.filterwarnings("ignore")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama3.2"
CACHE_DIR = Path(".cache_vectorstore")  # caché de índices FAISS en disco


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Página {i + 1} ---\n{page_text}"
    return text


# ─────────────────────────────────────────────
# 2. CHUNKING
# ─────────────────────────────────────────────


def split_text(
    text: str, chunk_size: int = 500, overlap: int = 100
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)


# ─────────────────────────────────────────────
# 3. EMBEDDINGS (se instancian una sola vez y se reutilizan)
# ─────────────────────────────────────────────


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─────────────────────────────────────────────
# 4. VECTORSTORE con caché en disco
#    Si el mismo PDF ya fue procesado, carga el índice guardado.
#    Si no, lo calcula y lo guarda para la próxima vez.
# ─────────────────────────────────────────────


def pdf_hash(pdf_path: str) -> str:
    """MD5 del archivo PDF para identificar si cambió."""
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_vectorstore(chunks: list[str], pdf_path: str) -> FAISS:
    embeddings = get_embeddings()
    cache_path = CACHE_DIR / pdf_hash(pdf_path)

    if cache_path.exists():
        print("        Índice encontrado en caché, cargando...")
        return FAISS.load_local(
            str(cache_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("        Generando embeddings (primera vez para este PDF)...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    CACHE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(cache_path))
    print("        Índice guardado en caché para próximas ejecuciones.")
    return vectorstore


# CARGA DE LLM LOCAL


def load_llm():
    print(f"        Cargando LLM: {LLM_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    try:
        # Se intenta cargar el modelo con cuantización INT8 usando bitsandbytes
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            LLM_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print("        Cuantización INT8 activa (modelo ~125 MB).")
    except Exception:
        # Fallback sin cuantización
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
        print("        Cuantización no disponible, cargando modelo completo.")

    model.eval()
    return model, tokenizer


def generate(prompt: str, model, tokenizer, max_new_tokens: int = 300) -> str:
    import torch

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    )
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# RESUMEN ESTRUCTURADO: se le pide al LLM que sintetice la información de manera clara y organizada.

SUMMARY_PROMPT = """\
Based on the following excerpts from a document, write a structured summary.
Include: (1) main topic, (2) up to 4 key points, (3) main conclusion.
Be concise.
 
Excerpts:
{context}
 
Structured summary:"""


def summarize(vectorstore: FAISS, model, tokenizer) -> str:
    # Recuperamos chunks representativos buscando temas generales
    queries = [
        "main topic and purpose of this document",
        "key arguments and findings",
        "conclusions and results",
    ]
    seen, docs = set(), []
    for q in queries:
        for doc in vectorstore.similarity_search(q, k=2):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc.page_content)

    context = "\n\n---\n\n".join(
        docs[:5]
    )  # máx 5 chunks para no superar 512 tokens
    return generate(SUMMARY_PROMPT.format(context=context), model, tokenizer)


QA_PROMPT = """\
Answer the question using ONLY the context below.
If the answer is not in the context, say: "I cannot find that information in the document."

Context:
{context}

Question: {question}

Answer:"""


def ask(question: str, vectorstore: FAISS, model, tokenizer) -> str:
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    return generate(
        QA_PROMPT.format(context=context, question=question), model, tokenizer
    )


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <ruta_al_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"ERROR: No se encuentra '{pdf_path}'")
        sys.exit(1)

    print(f"\n📄 Procesando: {pdf_path}\n")

    print("  [1/3] Extrayendo texto y preparando índice...")
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"        {len(raw_text)} caracteres extraídos.")
    chunks = split_text(raw_text)
    print(f"        {len(chunks)} chunks generados.")
    vectorstore = build_vectorstore(chunks, pdf_path)

    print("  [2/3] Cargando LLM (solo la primera vez tarda)...")
    model, tokenizer = load_llm()

    print("  [3/3] Generando resumen...\n")
    print("=" * 60)
    print("RESUMEN AUTOMÁTICO DEL DOCUMENTO")
    print("=" * 60)
    print(summarize(vectorstore, model, tokenizer))
    print("=" * 60)

    print("\nHaz preguntas sobre el documento (mejor en inglés).")
    print("Escribe 'salir' para terminar.\n")

    while True:
        question = input("Tu pregunta: ").strip()
        if question.lower() in ("salir", "exit", "quit"):
            print("¡Hasta luego!")
            break
        if not question:
            continue
        print(f"\n🤖 {ask(question, vectorstore, model, tokenizer)}\n")


if __name__ == "__main__":
    main()

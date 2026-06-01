import hashlib
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama3.2"
CACHE_DIR = Path(".cache_vectorstore")


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Página {i + 1} ---\n{page_text}"
    return text


def split_text(
    text: str, chunk_size: int = 800, overlap: int = 150
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)


# VECTORSTORE con caché en disco
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def pdf_hash(pdf_path: str) -> str:
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
            str(cache_path), embeddings, allow_dangerous_deserialization=True
        )

    print("        Generando embeddings (primera vez para este PDF)...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    CACHE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(cache_path))
    print("        Índice guardado en caché.")
    return vectorstore


def load_llm() -> OllamaLLM:
    print(f"        Conectando con Ollama: {LLM_MODEL}")
    return OllamaLLM(model=LLM_MODEL, temperature=0.01)


SUMMARY_PROMPT = """\
You are an expert document analyst. Based on the following excerpts from a document, write a structured summary in the same language as the text.

Include:
1. Main topic
2. Key points (up to 5)
3. Main conclusion

Excerpts:
{context}

Structured summary:"""


def summarize(vectorstore: FAISS, llm: OllamaLLM) -> str:
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

    context = "\n\n---\n\n".join(docs[:6])
    return llm.invoke(SUMMARY_PROMPT.format(context=context))


QA_PROMPT = """\
You are an expert document analyst. Answer the question using ONLY the context below.
Answer in the same language as the question.
If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""


def ask(question: str, vectorstore: FAISS, llm: OllamaLLM) -> str:
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)
    return llm.invoke(QA_PROMPT.format(context=context, question=question))


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

    print("  [2/3] Cargando LLM via Ollama...")
    llm = load_llm()

    print("  [3/3] Generando resumen...\n")
    print("=" * 60)
    print("RESUMEN AUTOMÁTICO DEL DOCUMENTO")
    print("=" * 60)
    print(summarize(vectorstore, llm))
    print("=" * 60)

    print("\nHaz preguntas sobre el documento.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        question = input("Tu pregunta: ").strip()
        if question.lower() in ("salir", "exit", "quit"):
            print("¡Hasta luego!")
            break
        if not question:
            continue
        print(f"\n🤖 {ask(question, vectorstore, llm)}\n")


if __name__ == "__main__":
    main()

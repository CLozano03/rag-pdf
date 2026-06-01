# RAG PDF Q&A

Sistema de resumen y preguntas sobre documentos PDF usando RAG.

## Requisitos
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) instalado
- Ollama (para el modelo LLaMA 3.2)

## Instalación rápida

```bash

# Instalar Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Si quiero empezar el servidor manualmente
ollama serve

# Parar el servicio
sudo systemctl stop ollama

# Para comprobar que esta activo
curl http://localhost:11434 


\begin{lstlisting}[language=bash]
# Descargar el modelo (~2 GB, solo la primera vez)
ollama pull llama3.2

# Arrancar el servidor (queda escuchando en :11434)
ollama serve
\end{lstlisting}
```

Una vez que el servidor de Ollama esté corriendo, puedes instalar las dependencias y ejecutar el script de Python:


```bash
# Si no tienes uv instalado, puedes instalarlo asi:
curl -fsSL https://install.astral.sh/uv/install.sh | sh

# Instalar dependencias con uv
uv sync

# Ejecutar
uv run python main.py <tu_documento.pdf>
```

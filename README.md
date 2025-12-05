<p align="center">
  <img src="https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/banner.png" width="100%" alt="ChefSense AI Banner">
</p>

# 🍳 ChefSense AI — Adam Ragusea–Style Cooking Assistant  
*A multimodal RAG system using YouTube transcripts, LangChain, Whisper, ReAct Agents & LangSmith*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" />
  <img src="https://img.shields.io/badge/Framework-LangChain-green" />
  <img src="https://img.shields.io/badge/Database-ChromaDB-purple" />
  <img src="https://img.shields.io/badge/Model-gpt4o--mini-orange" />
  <img src="https://img.shields.io/badge/UI-Gradio-yellow" />
  <img src="https://img.shields.io/badge/Eval-LangSmith-lightgrey" />
</p>

---

## 🧭 Overview

ChefSense AI is an experimental cooking assistant that answers questions in the style of **Adam Ragusea** — known for his calm tone, science-based cooking explanations, and “on the one hand / on the other hand” trade-off reasoning.

The system:
- Ingests **YouTube cooking videos**
- Extracts transcripts using **Whisper**
- Stores transcript chunks in **ChromaDB**
- Retrieves relevant segments using **RAG**
- Uses a **ReAct Agent** with a custom persona prompt
- Produces **Adam-style text + voice answers**
- Supports **text chat + voice chat** in a custom-designed Gradio UI

---

# 🎥 Video Demo

# 🏗️ Architecture

User Input (Text or Voice)
↓
Gradio UI
↓
ReAct Agent ←—— Memory
↓
Tools (Retriever / Full Transcript)
↓
Chroma Vector DB
↓
Transcript Chunks
↓
RAG Prompt + Persona
↓
OpenAI LLM (gpt-4o-mini)
↓
Adam-Style Output (Text + Optional Audio)


---

# 📦 Tech Stack

| Component | Technology |
|----------|------------|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | `text-embedding-3-small` |
| Vector DB | ChromaDB |
| Audio Transcription | Whisper |
| Video Download | yt-dlp |
| Framework | LangChain (retrievers, agents, prompts) |
| UI | Gradio Blocks |
| Evaluation | Manual + LangSmith criteria evaluation |

---

# 📼 Dataset & Ingestion Pipeline

## 📥 Data Sources
- YouTube cooking videos (Adam Ragusea)  
- Downloaded with **yt-dlp**

## 🎧 Transcription (Whisper)

```python
whisper_model = whisper.load_model("small")
result = whisper_model.transcribe(audio_path)
```


## 🗃 Storage

Raw transcripts saved under data/raw_transcripts/

Metadata stored as:

json

{ "source": "abc123.txt", "video_id": "abc123" }
🔪 Preprocessing
Chunked into ~800 tokens with 200 overlap

Embedded using OpenAI embeddings

🧩 **Vector DB Build** 
python

vectordb = Chroma.from_documents(
    chunks,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="data/chroma_db"
)

## 🧠 Persona & Prompt Engineering
**Adam Persona (Excerpt)**
pgsql

You are an unofficial Adam Ragusea–style cooking assistant.
Always start with a Transcript Check.
If transcript is missing, state it clearly.
Use “On the one hand…” / “On the other hand…” for trade-offs.
Never invent transcript content.
Maintain Adam’s calm, science-forward tone.

**RAG Prompt Template**
python

ADAM_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ADAM_PERSONA),
    ("system", "Use the transcript context:\n{context}"),
    ("human", "{input}")
])

**ReAct Agent Prompt**
python
Copy code
react_prompt = ChatPromptTemplate.from_messages([
    ("system", ADAM_PERSONA + "\nYou have access to:\n{tools}"),
    ("system", "Use tools when needed. Think step-by-step."),
    ("human", "{input}")
])

## 🧪EVALUATION
**✔️ Manual Evaluation**
20 custom questions evaluating:
Helpfulness

Food Safety

Adam-Style Alignment

Examples:
“In the risotto video, why does Adam stir so much?”

“Is it safe to store cooked chicken for five days?”

“Should I rinse rice before cooking it?”

“What’s the trade-off between butter and oil for steak?”

Scoring
Annotated CSV → crosstabs:

Helpfulness × Style

Safety × Helpfulness

**✔️ LangSmith Evaluation**
Criteria Evaluated
python
Copy code
criteria = {
  "helpfulness": "...",
  "food_safety": "...",
  "adam_style": "..."
}
Target Function
python

def chef_bot_predict(inputs):
    chain_input = {"input": inputs["question"], "chat_history": []}
    return {"output": chain.invoke(chain_input)}
    
Why LangSmith?
Automated criteria-based scoring

Trace-level debugging

Reproducible experiments

Comparison against expected notes

## 🖥️ Running the App
Launch from Jupyter Notebook:

python

if __name__ == "__main__":
    init_vectordb_and_chains()
    app = build_gradio_app()
    app.launch()
Features:

YouTube ingestion

Text chat

Voice chat

Voice output

## 📁 Project Structure
css

project/
│
├── data/
│   ├── raw_transcripts/
│   ├── chroma_db/
│   └── audio/
│
├── generated_audio/
├── chefsense.ipynb
├── demo/
│   └── demo-preview.gif
│
├── chef_sense_manual_eval_template.csv
├── chef_sense_manual_evaluated.csv
├── README.md
└── requirements.txt


## 🧰 Installation
bash

git clone https://github.com/YOUR_USERNAME/chefsense-ai
cd chefsense-ai
pip install -r requirements.txt
Set environment variables:

ini

OPENAI_API_KEY=your_api_key
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_PROJECT=ChefSense


## 🛣️ Future Work

Deployment (FastAPI + HuggingFace Spaces)

Multi-speaker diarization

Improved retrieval filtering

Concise answer tuning

Multi-language support

Structured recipe generation mode with image retrieval

Improved UI

## 🧑‍🍳 Credits
ChefSense AI is an academic/demo project and is not affiliated with Adam Ragusea or YouTube.
All media is used purely for research and educational purposes.


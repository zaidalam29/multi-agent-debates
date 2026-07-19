# ⚖️ Multi-Agent Debate System

Two AI agents debate any topic from opposing sides, with an impartial AI judge scoring the arguments and declaring a winner — built with a multi-agent orchestration pattern and automatic LLM failover.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Multi--Agent-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai&logoColor=white)](https://platform.openai.com/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Fallback-6E56CF?style=flat)](https://openrouter.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

**Author**: Zaid Alam — AI Engineer

---

## ✨ Features

- **Two opposing AI debaters** with distinct expert personas (PRO / CON)
- **Impartial AI judge** that scores both sides and synthesizes a verdict
- **Multi-round debates** — each agent rebuts the opponent's previous argument
- **Automatic LLM failover** — falls back from OpenAI to OpenRouter on quota, rate-limit, or auth errors, with zero manual intervention
- **Configurable** topic and round count via CLI flags

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| Orchestration | LangChain (`langchain-core`, `langchain-openai`) |
| Primary LLM | OpenAI GPT-4o-mini (debaters), GPT-4o (judge) |
| Fallback LLM | OpenRouter (OpenAI-compatible endpoint) |
| Config | python-dotenv |
| Interface | CLI (argparse) |

## 🏗️ Architecture

```
                     ┌────────────────────┐
   --topic  ───────► │   run_debate()      │
                     └─────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                  ▼
     ┌────────────────┐                 ┌────────────────┐
     │   PRO Agent     │◄──rebuts───────►│   CON Agent     │
     │  (GPT-4o-mini)  │   N rounds      │  (GPT-4o-mini)  │
     └────────┬────────┘                 └────────┬────────┘
              │                                    │
              └────────────────┬───────────────────┘
                                ▼
                       ┌─────────────────┐
                       │   Judge Agent    │
                       │    (GPT-4o)      │
                       └────────┬─────────┘
                                ▼
                          Final Verdict
                  (winner, scores, synthesis)

  Every LLM call ─► try OpenAI ─► fails? ─► auto-retry on OpenRouter
```

Each debater keeps its own argument history; the judge receives the full transcript from both sides at the end and returns a structured verdict — winner, score out of 10 per side, strongest argument per side, and a balanced synthesis.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/multi-agent-debates.git
cd multi-agent-debates
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...          # optional — only needed if you have OpenAI quota
OPENROUTER_API_KEY=sk-or-...   # get a free key at https://openrouter.ai/keys
```

> If `OPENAI_API_KEY` is missing or fails (quota/auth/rate-limit), the app automatically switches to OpenRouter using the same conversation — no code changes needed.

### 5. Run a debate

```bash
python agent.py --topic "Universal Basic Income should replace traditional welfare systems"
python agent.py --topic "Social media does more harm than good to society" --rounds 3
python agent.py --topic "Nuclear energy is the best solution to climate change" --rounds 2
```

| Flag | Description | Default |
|---|---|---|
| `--topic` | The debate topic | `"AI will create more jobs than it eliminates over the next decade"` |
| `--rounds` | Number of debate rounds (1–4) | `2` |

## 📁 Project Structure

```
multi-agent-debates/
├── agent.py            # Debaters, judge, orchestration, LLM fallback logic
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── README.md
```

## 🛣️ Roadmap

- [ ] Streamlit / web UI for live debate viewing
- [ ] Export verdict + transcript as PDF
- [ ] Support for more than 2 debaters (panel mode)
- [ ] Pluggable local model support (Ollama)

## 📄 License

MIT

---

Built by **Zaid Alam** — AI Engineer
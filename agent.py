"""
Multi-Agent Debate System using AutoGen-style orchestration.

Two AI agents debate a topic from opposing sides, moderated by a judge
who declares a winner and synthesizes the key arguments.

Fallback: if the OpenAI call fails (quota / rate-limit / auth error),
it automatically switches to OpenRouter with the same messages.

Usage:
    python agent.py --topic "AI will eliminate more jobs than it creates"
    python agent.py --topic "Remote work is better than office work" --rounds 3

Add to .env:
    OPENAI_API_KEY=sk-...          (optional, only needed if you have quota)
    OPENROUTER_API_KEY=sk-or-...   (get one at https://openrouter.ai/keys)
"""

import argparse
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIError, APIStatusError, RateLimitError

load_dotenv()

# Errors that should trigger a fallback to OpenRouter
FALLBACK_ERRORS = (RateLimitError, APIStatusError, APIError)

# Model mapping for OpenRouter (OpenAI-compatible endpoint)
OPENROUTER_MODEL_MAP = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o": "openai/gpt-4o",
}
OPENROUTER_FREE_FALLBACK = "meta-llama/llama-3.1-8b-instruct:free"

# Keep this low so free/low-credit OpenRouter accounts don't get a 402
# ("requires more credits, or fewer max_tokens") error.
MAX_TOKENS = 400


def make_llm(openai_model: str, temperature: float) -> ChatOpenAI:
    """Primary LLM is OpenAI. If OPENAI_API_KEY is missing, go straight to OpenRouter."""
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model=openai_model, temperature=temperature, max_tokens=MAX_TOKENS)
    return make_openrouter_llm(openai_model, temperature)


def make_openrouter_llm(openai_model: str, temperature: float) -> ChatOpenAI:
    """OpenRouter exposes an OpenAI-compatible API, so ChatOpenAI works as-is."""
    or_key = os.getenv("OPENROUTER_API_KEY")
    if not or_key:
        raise RuntimeError(
            "OpenAI call failed and OPENROUTER_API_KEY is not set in .env. "
            "Get a free key at https://openrouter.ai/keys"
        )
    or_model = OPENROUTER_MODEL_MAP.get(openai_model, OPENROUTER_FREE_FALLBACK)
    return ChatOpenAI(
        model=or_model,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        api_key=or_key,
        base_url="https://openrouter.ai/api/v1",
    )


def invoke_with_fallback(primary_llm, fallback_llm_factory, messages):
    """Try the primary LLM first; fall back to OpenRouter on a known error."""
    try:
        return primary_llm.invoke(messages)
    except FALLBACK_ERRORS as e:
        print(f"Primary LLM failed ({type(e).__name__}), switching to OpenRouter...")
        fallback_llm = fallback_llm_factory()
        return fallback_llm.invoke(messages)


class DebateAgent:
    def __init__(self, name: str, position: str, expertise: str):
        self.name = name
        self.position = position
        self.expertise = expertise
        self.model_name = "gpt-4o-mini"
        self.temperature = 0.6
        self.llm = make_llm(self.model_name, self.temperature)
        self.arguments = []

    def make_argument(self, topic: str, round_num: int, opponent_last_arg: str = "") -> str:
        system_msg = f"""You are {self.name}, a {self.expertise}.
You are arguing {self.position} on this topic.
Make compelling, evidence-based arguments. Be direct and persuasive.
Keep response under 150 words. Round {round_num}."""

        if opponent_last_arg:
            user_msg = f"Topic: {topic}\n\nYour opponent just said: '{opponent_last_arg}'\n\nRespond and advance your argument:"
        else:
            user_msg = f"Topic: {topic}\n\nMake your opening argument for {self.position.upper()}:"

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ]

        response = invoke_with_fallback(
            self.llm,
            lambda: make_openrouter_llm(self.model_name, self.temperature),
            messages,
        )
        argument = response.content
        self.arguments.append(argument)
        return argument


class DebateJudge:
    def __init__(self):
        self.model_name = "gpt-4o"
        self.temperature = 0
        self.llm = make_llm(self.model_name, self.temperature)

    def evaluate(self, topic: str, pro_agent: DebateAgent, con_agent: DebateAgent) -> dict:
        pro_args = "\n\n".join(f"Round {i+1}: {a}" for i, a in enumerate(pro_agent.arguments))
        con_args = "\n\n".join(f"Round {i+1}: {a}" for i, a in enumerate(con_agent.arguments))

        messages = [
            SystemMessage(content="""You are an impartial debate judge. Evaluate both sides fairly.
Return a structured verdict with: winner, score (out of 10 each), strongest argument per side, key insights, and balanced synthesis conclusion."""),
            HumanMessage(content=f"""Topic: "{topic}"

PRO arguments ({pro_agent.name}):
{pro_args}

CON arguments ({con_agent.name}):
{con_args}

Provide your verdict:"""),
        ]

        response = invoke_with_fallback(
            self.llm,
            lambda: make_openrouter_llm(self.model_name, self.temperature),
            messages,
        )
        return {"verdict": response.content}


def run_debate(topic: str, rounds: int = 2) -> None:
    pro = DebateAgent(
        name="Dr. Abdul Kalam",
        position="FOR",
        expertise="technology economist and AI researcher"
    )
    con = DebateAgent(
        name="Prof. Zakir",
        position="AGAINST",
        expertise="labor economist and social policy expert"
    )
    judge = DebateJudge()

    print(f"\n{'='*60}")
    print(f"DEBATE: {topic}")
    print(f"{'='*60}")
    print(f"FOR: {pro.name} ({pro.expertise})")
    print(f"AGAINST: {con.name} ({con.expertise})")
    print(f"Rounds: {rounds}")
    print("=" * 60)

    last_con_arg = ""
    last_pro_arg = ""

    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num} ---\n")

        pro_arg = pro.make_argument(topic, round_num, last_con_arg)
        print(f"{pro.name} (FOR):")
        print(pro_arg)

        con_arg = con.make_argument(topic, round_num, pro_arg)
        print(f"\n{con.name} (AGAINST):")
        print(con_arg)

        last_pro_arg = pro_arg
        last_con_arg = con_arg

    print(f"\n{'='*60}")
    print("JUDGE'S VERDICT")
    print("=" * 60)
    verdict = judge.evaluate(topic, pro, con)
    print(verdict["verdict"])


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Debate System")
    parser.add_argument("--topic", default="AI will create more jobs than it eliminates over the next decade", help="Debate topic")
    parser.add_argument("--rounds", type=int, default=2, help="Number of debate rounds (1-4)")
    args = parser.parse_args()

    rounds = max(1, min(4, args.rounds))
    run_debate(args.topic, rounds)


if __name__ == "__main__":
    main()
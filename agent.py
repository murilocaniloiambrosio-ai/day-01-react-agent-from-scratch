"""
A minimal ReAct-style AI agent, built from scratch in Python.

No framework, no external API calls. The "LLM" is mocked by a deterministic
function (`fake_llm`) that inspects the user's question and decides which
tool to call next -- the exact same job a real LLM does when it receives
a prompt with the list of available tools. Everything else (the loop,
the tool dispatch, the trace printing) is exactly how a real agent works.

Loop:
    1. Thought  -> the "model" reasons about what to do next
    2. Action   -> the "model" picks a tool + input
    3. Observation -> the agent runs the tool and returns the result as text
    4. repeat until the "model" says Final Answer
"""

import re


# ---------------------------------------------------------------------------
# 1. Tools
#    Each tool is just a plain Python function. In a real agent, the model
#    only sees the tool's name + description (like a function signature) and
#    picks one; the actual execution always happens in our code, never inside
#    the model.
# ---------------------------------------------------------------------------

def tool_calculator(expression: str) -> str:
    """Evaluates a basic arithmetic expression, e.g. '12 * 7 + 1'."""
    # NOTE: eval() is used here only because this is a controlled learning
    # exercise with a mocked model. A production agent would use a safe
    # expression parser (e.g. `ast.literal_eval` won't work for math, so
    # a real project should use a library like `simpleeval`).
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression))
    except Exception as exc:
        return f"Error: {exc}"


def tool_fake_weather(city: str) -> str:
    """Returns a fake weather report for a given city (no real API call)."""
    fake_db = {
        "guarulhos": "24C, partly cloudy",
        "sao paulo": "21C, light rain",
        "new york": "18C, windy",
        "london": "15C, overcast",
    }
    key = city.strip().lower()
    return fake_db.get(key, f"No weather data for '{city}' (mock database).")


def tool_text_search(query: str) -> str:
    """Searches a small in-memory knowledge base for a keyword."""
    knowledge_base = [
        "ReAct combines Reasoning and Acting in a single loop.",
        "Tool use lets a model call external functions to get real data.",
        "DEVFORGE is Caniloi's personal software/design brand, cyberpunk/neon style, accent color #00CFFF.",
    ]
    # Naive substring matching on the whole query fails for real questions
    # (e.g. "what is ReAct?" never appears verbatim in the knowledge base).
    # Splitting into keywords and matching any of them is closer to how a
    # real retrieval step behaves.
    keywords = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
    matches = [line for line in knowledge_base if any(kw in line.lower() for kw in keywords)]
    return " | ".join(matches) if matches else "No matching entry found."


TOOLS = {
    "calculator": tool_calculator,
    "weather": tool_fake_weather,
    "search": tool_text_search,
}


# ---------------------------------------------------------------------------
# 2. Mocked "LLM"
#    A real agent would send the question + tool list + conversation history
#    to an LLM API and parse a response like:
#        Thought: I need to calculate this.
#        Action: calculator
#        Action Input: 12 * 7
#    Here we simulate exactly that output format using simple keyword rules.
# ---------------------------------------------------------------------------

def fake_llm(question: str, history: list) -> str:
    """
    Simulates one LLM turn. Returns text formatted exactly like a real
    model's ReAct-style response, so the parsing logic below works
    identically to how it would with a real API.
    """
    last_observation = history[-1]["observation"] if history else None

    # If we already have an observation, decide whether it's enough to answer.
    if last_observation is not None:
        return f"Thought: I now have enough information.\nFinal Answer: {last_observation}"

    # Otherwise, decide which tool to use based on the question content.
    math_match = re.search(r"[\d()+\-*/. ]{3,}", question)
    if any(word in question.lower() for word in ["how much", "calculate", "sum", "+", "*"]) and math_match:
        expr = math_match.group().strip()
        return f"Thought: This looks like a math question, I should use the calculator.\nAction: calculator\nAction Input: {expr}"

    if "weather" in question.lower():
        city_match = re.search(r"in ([a-zA-Z ]+)", question)
        city = city_match.group(1).strip() if city_match else "guarulhos"
        return f"Thought: This is a weather question, I should check the weather tool.\nAction: weather\nAction Input: {city}"

    return f"Thought: I should search my knowledge base for this.\nAction: search\nAction Input: {question}"


# ---------------------------------------------------------------------------
# 3. The agent loop
# ---------------------------------------------------------------------------

def parse_response(text: str):
    """Extracts action/action_input or final_answer from the model's text."""
    if "Final Answer:" in text:
        return {"final_answer": text.split("Final Answer:")[-1].strip()}

    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(.+)", text)
    return {
        "action": action_match.group(1).strip() if action_match else None,
        "action_input": input_match.group(1).strip() if input_match else None,
    }


def run_agent(question: str, max_steps: int = 4) -> str:
    print(f"\n=== New question: {question!r} ===")
    history = []

    for step in range(1, max_steps + 1):
        print(f"\n-- Step {step} --")
        response_text = fake_llm(question, history)
        print(response_text)

        parsed = parse_response(response_text)

        if "final_answer" in parsed:
            print(f"\n>>> Agent finished after {step} step(s).")
            return parsed["final_answer"]

        action, action_input = parsed["action"], parsed["action_input"]
        if action not in TOOLS:
            return f"Error: model requested unknown tool '{action}'."

        observation = TOOLS[action](action_input)
        print(f"Observation: {observation}")

        history.append({"action": action, "action_input": action_input, "observation": observation})

    return "Error: max steps reached without a final answer."


if __name__ == "__main__":
    questions = [
        "How much is 12 * 7 + 1?",
        "What's the weather in Guarulhos?",
        "What is ReAct?",
    ]
    for q in questions:
        answer = run_agent(q)
        print(f"\nFINAL ANSWER: {answer}")
        print("=" * 60)

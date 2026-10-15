"""MBPP prompting (few-shot style close to Austin et al.)."""

from __future__ import annotations

from rlvr.data.mbpp import MBPPProblem


def format_tests(problem: MBPPProblem) -> str:
    return "\n".join(problem.test_list)


def format_example(problem: MBPPProblem, include_solution: bool) -> str:
    body = (
        f"You are an expert Python programmer, and here is your task: {problem.text} "
        f"Your code should pass these tests:\n\n{format_tests(problem)}\n[BEGIN]\n"
    )
    if include_solution:
        return body + f"{problem.code.rstrip()}\n[DONE]"
    return body


def build_prompt(problem: MBPPProblem, few_shot: list[MBPPProblem]) -> str:
    parts = [format_example(ex, include_solution=True) for ex in few_shot]
    parts.append(format_example(problem, include_solution=False))
    return "\n".join(parts)


def extract_code(completion: str) -> str:
    """Extract program text from a model completion."""
    text = completion.strip()
    if "[DONE]" in text:
        text = text.split("[DONE]", 1)[0]
    if "[BEGIN]" in text:
        text = text.split("[BEGIN]", 1)[-1]
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

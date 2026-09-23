"""
Mutators backed by an LLM: one rewrites the painter's prompt, one redesigns the brush set.

Both force a single tool call so the reply is structured. A reply that can't be used raises, which
ObservedMutator records as a mutator error instead of silently producing nothing.
"""

from __future__ import annotations

import re

from darwinian_evolver.problem import Mutator
from pydantic import ValidationError

from conveyor.llm import LLMClient
from conveyor.llm import function_tool
from conveyor.painting.brushcode import SIGNATURE
from conveyor.painting.brushcode import render_source
from conveyor.painting.canvas import LIMITS
from conveyor.painting.canvas import MAX_BRUSHES
from conveyor.painting.canvas import Brush
from conveyor.painting.llm_agent import PromptStrategy
from conveyor.painting.toolkit import Toolkit

MAX_PROMPT_CHARS = 1500


def _failures_text(failures) -> str:
    lines = []
    for f in failures:
        lines.append(
            f"- {f.data_point_id}, {f.failure_type.replace('_', ' ')}: the best possible with these brushes has "
            f"error {f.oracle_err:.3f}, the painting has {f.agent_err:.3f}"
        )
    return "\n".join(lines)


def _log_text(entries) -> str:
    if not entries:
        return "None yet."
    return "\n".join(f"- Tried: {e.attempted_change}\n  Result: {e.observed_outcome}" for e in entries[:8])


def _forced(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def _outcome(client: LLMClient, reply, status: str, note: str) -> None:
    """Record on the call's row what became of its tool calls: the first matching call is the one used."""
    results = [{"status": "ignored", "error": "not used"} for _ in reply.tool_calls]
    if results:
        results[0] = {"status": status, ("note" if status == "applied" else "error"): note}
    client.annotate(reply.call_id, tool_results=results)


def _find_call(reply, name: str) -> dict:
    for call in reply.tool_calls:
        if call.name == name and call.arguments is not None:
            return call.arguments
    raise ValueError(f"model did not call {name} with valid arguments (finish_reason={reply.finish_reason})")


class LLMPromptMutator(Mutator):
    """Rewrites the painter's instructions to address patches the painting got wrong."""

    TOOL = function_tool(
        "revise_prompt",
        "Submit the revised painter instructions.",
        {
            "prompt": {"type": "string", "description": f"The full new instructions, under {MAX_PROMPT_CHARS} characters."},
            "summary": {"type": "string", "description": "One sentence: what you changed and why."},
        },
        ["prompt", "summary"],
    )

    def __init__(self, client: LLMClient) -> None:
        super().__init__()
        self.client = client

    @property
    def supports_batch_mutation(self) -> bool:
        return True

    def mutate(self, organism: PromptStrategy, failure_cases, learning_log_entries):
        system = (
            "You improve the instructions given to an AI painter. The painter copies a target painting onto a "
            "simulated watercolor canvas by calling brush tools, looking at the target and its canvas each turn. "
            "The set of brushes changes over time, so never refer to a brush by name. Describe strategy: order of "
            "work, where to place strokes, how to pick colors and directions, when to switch to fine brushes."
        )
        user = (
            f"Current instructions:\n<<<\n{organism.prompt.strip()}\n>>>\n\n"
            "The harness sets the stroke budget, so keep stroke counts and budgets out of the instructions "
            "(one rewrite told the painter to 'keep 500 strokes in reserve', which was its whole budget). "
            f"These regions came out wrong even though the brushes "
            f"could have done better, so the painter is at fault:\n{_failures_text(failure_cases)}\n\n"
            f"Earlier changes to these instructions and what they did:\n{_log_text(learning_log_entries)}\n\n"
            "Revise the instructions to fix these failures. Don't repeat a change that made things worse."
        )
        reply = self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            purpose="mutate",
            tools=[self.TOOL],
            tool_choice=_forced("revise_prompt"),
            max_tokens=4096,
        )
        try:
            args = _find_call(reply, "revise_prompt")
            prompt = str(args.get("prompt", "")).strip()[:MAX_PROMPT_CHARS]
            if not prompt:
                raise ValueError("model returned an empty prompt")
        except ValueError as e:
            _outcome(self.client, reply, "rejected", str(e))
            raise
        if prompt == organism.prompt.strip():
            _outcome(self.client, reply, "ignored", "same as the current prompt")
            return []
        _outcome(self.client, reply, "applied", "became a new prompt organism")
        summary = str(args.get("summary", "")).strip() or "Revised the instructions."
        return [
            PromptStrategy(
                prompt=prompt,
                n_strokes=organism.n_strokes,
                strokes_per_turn=organism.strokes_per_turn,
                from_change_summary=f"[llm] {summary}",
            )
        ]


def _clean_name(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")[:40] or "brush"
    if base[0].isdigit():
        base = "b_" + base
    candidate, i = base, 2
    while candidate in taken:
        candidate = f"{base}_{i}"
        i += 1
    taken.add(candidate)
    return candidate


class LLMToolkitMutator(Mutator):
    """Redesigns the brush set for patches that no stroke sequence with the current brushes can reproduce."""

    @staticmethod
    def _tool() -> dict:
        """Built per call, not at import: the size limits scale with the canvas, which the graph sets on build."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short, lowercase, no spaces."},
                **{attr: {"type": "number", "minimum": lo, "maximum": hi} for attr, (lo, hi) in LIMITS.items()},
                "doc": {"type": "string", "description": "One short line: what the mark looks like. The painter reads this."},
                "source": {
                    "type": "string",
                    "description": f"Python defining `{SIGNATURE}`, returning the alpha mask. See the rules.",
                },
            },
            "required": ["name", *LIMITS.keys(), "doc", "source"],
        }
        return function_tool(
            "revise_toolkit",
            "Submit the complete revised brush set.",
            {
                "brushes": {"type": "array", "items": schema, "maxItems": MAX_BRUSHES},
                "summary": {"type": "string", "description": "One sentence: what you changed and why."},
            },
            ["brushes", "summary"],
        )

    def __init__(self, client: LLMClient) -> None:
        super().__init__()
        self.client = client

    @property
    def supports_batch_mutation(self) -> bool:
        return True

    def mutate(self, organism: Toolkit, failure_cases, learning_log_entries):
        limits = "\n".join(f"- {attr}: {lo} to {hi}" for attr, (lo, hi) in LIMITS.items())
        system = (
            "You design brushes for a painting simulator, and each brush is a small Python program that draws "
            "its own mark. A painter uses them to copy paintings by Vincent van Gogh, whose canvases mix broad "
            "directional sweeps with short ridged strokes that follow the form.\n\n"
            f"Each brush gives `radius` (how far the mark reaches from the stroke path), `length` (the path, 0 "
            f"for a dab), and `source`, which defines:\n\n    {SIGNATURE}:\n\n"
            "`u` is distance along the path, 0 at the start and `length` at the end. `v` is signed distance "
            "across it. Both are float32 arrays of the same shape, in pixels, covering a box that reaches "
            "`radius` past the path on every side. Return an array of that same shape, 0 to 1, saying how much "
            "pigment lands at each pixel. `rng` is a numpy Generator, for grain and scatter.\n\n"
            "The code runs in a sandbox: no imports, no `while`, no names starting with an underscore, and "
            "nothing in scope but `np`, `math`, `rng`, the arguments, and simple builtins. A brush that fails "
            "to compile, returns the wrong shape, goes outside 0 to 1, or lays no pigment makes the whole "
            "toolkit non-viable.\n\n"
            "You are not limited to the marks already in the set. Modulating on `u` breaks a line into dots; "
            "modulating on `v` gives parallel bristles; scaling the reach by `u / length` tapers the stroke. "
            "Here is a plain one to work from:\n\n"
            f"{render_source('capsule', {})}"
        )
        current = "\n\n".join(
            f"- {b.name} (radius {b.radius}, length {b.length}): {b.doc or 'no description'}\n{b.source.rstrip()}"
            for b in organism.brushes
        )
        user = (
            f"Current brushes:\n\n{current}\n\nLimits (a toolkit outside them is rejected), at most {MAX_BRUSHES} "
            f"brushes:\n{limits}\n\nThese regions can't be reproduced even by the best possible strokes with the current "
            f"brushes, so the brush set is at fault:\n{_failures_text(failure_cases)}\n\n"
            f"Earlier changes to the brush set and what they did:\n{_log_text(learning_log_entries)}\n\n"
            "Return the full revised brush set. Keep brushes that work. Change as little as fixes the failures."
        )
        reply = self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            purpose="mutate",
            tools=[self._tool()],
            tool_choice=_forced("revise_toolkit"),
            max_tokens=4096,
        )
        try:
            args = _find_call(reply, "revise_toolkit")
            raw = args.get("brushes")
            if not isinstance(raw, list) or not raw:
                raise ValueError("model returned no brushes")
            taken: set[str] = set()
            brushes = []
            for item in raw:
                if not isinstance(item, dict):
                    raise ValueError(f"brush entry is not an object: {item!r:.80}")
                try:
                    fields = {k: float(item[k]) for k in LIMITS}
                    source = str(item["source"])
                    brushes.append(Brush(
                        name=_clean_name(str(item.get("name", "brush")), taken),
                        doc=str(item.get("doc", "")).strip()[:120],
                        source=source,
                        **fields,
                    ))
                except (KeyError, TypeError, ValueError, ValidationError) as e:
                    raise ValueError(f"bad brush {item!r:.120}: {e}") from e
        except ValueError as e:
            _outcome(self.client, reply, "rejected", str(e))
            raise
        _outcome(self.client, reply, "applied", f"became a toolkit with {len(brushes)} brushes")
        summary = str(args.get("summary", "")).strip() or "Revised the brush set."
        # Out-of-limit values and brush code that will not compile pass through on purpose: the evaluator
        # marks the toolkit non-viable and the dashboard shows why, which says more about the mutator than
        # dropping the child here would.
        return [Toolkit(brushes=brushes, from_change_summary=f"[llm] {summary}")]

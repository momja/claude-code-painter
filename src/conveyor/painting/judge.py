"""
A model judge for finished paintings: a model looks at the target and the painting and scores the copy on a rubric.

The deterministic critic scores pixels and texture statistics. Against a person's ratings its texture half did
well and its pixel half poorly, and a small vision model (Laya) could not rank the paintings at all. A large
model with real knowledge of painting can judge what the proxy can't: whether the face reads as that face,
whether the brushwork moves the way van Gogh's does.

The judge is blind. It sees the target and one painting, never which organism made it, its parent, or the
champion. It scores the finished painting only; the painter keeps the instant numeric critic for its per-call
feedback, so it can't probe the judge one stroke at a time. Each verdict carries a critique, which goes to the
mutators alongside the painter's own note.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from conveyor.harness import Job
from conveyor.harness import ProcessHarness
from conveyor.store import new_id

CRITERIA = ("likeness", "colour", "brushwork", "overall")

SYSTEM = """\
You judge copies of a painting. You are shown the original and one copy, both at the same small size. Score the \
copy as a demanding painting teacher would, on four criteria, each an integer from 1 to 10:

- likeness: does it show the same subject, recognisably? The pose, the face and its features, the placement of \
the figure, the main shapes.
- colour: are the colours and the lights and darks where they are in the original?
- brushwork: is the paint handled the way the original's is? For van Gogh, that means directional, rhythmic \
strokes that follow the forms, broken colour, visible texture; not smooth fills, blur, or random scribble.
- overall: how good a copy of this painting it is, all things considered.

Anchors, the same for every criterion: 10 is indistinguishable from the original at this size; 7 is a skilled \
copy with clear flaws; 5 is recognisably the same picture, done crudely; 3 is a loose resemblance at best; 1 \
has nothing to do with it. Use the whole range. A blurred or flat copy has poor brushwork however close its \
colours are, and a textured copy of the wrong picture has poor likeness however painterly it looks.

Then write a critique for the person trying to improve the copy: two to four sentences, the most important \
problem first, naming where on the canvas it is. Say what to change, not just what is wrong."""

SCHEMA = {
    "type": "object",
    "properties": {
        **{c: {"type": "integer", "minimum": 1, "maximum": 10} for c in CRITERIA},
        "critique": {"type": "string"},
    },
    "required": [*CRITERIA, "critique"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    scores: dict[str, int]
    critique: str
    session_id: str | None
    cost: float

    @property
    def score(self) -> float:
        """0 to 1: the mean of the four criteria, less noisy than `overall` alone."""
        return (sum(self.scores[c] for c in CRITERIA) / len(CRITERIA) - 1) / 9


class JudgeError(RuntimeError):
    pass


class ClaudeJudge:
    def __init__(self, claude: ProcessHarness, work_dir: Path, budget_usd: float = 1.0) -> None:
        self.claude = claude
        self.work_dir = Path(work_dir)
        self.budget_usd = budget_usd

    def judge(self, target_png: bytes, painting_png: bytes, *, organism_id: str | None = None) -> Verdict:
        outcome = self.claude.run(Job(
            purpose="judge", system_prompt=SYSTEM,
            content=[{"type": "text", "text": "The original:"}, {"type": "png", "data": target_png},
                     {"type": "text", "text": "The copy to judge:"}, {"type": "png", "data": painting_png}],
            cwd=self.work_dir / f"judge-{new_id()}", json_schema=SCHEMA, max_budget_usd=self.budget_usd,
            node="judge", organism_id=organism_id))
        out = outcome.structured if isinstance(outcome.structured, dict) else _json_in(outcome.result)
        if not out or any(not isinstance(out.get(c), int | float) for c in CRITERIA):
            raise JudgeError(f"the judge returned no usable verdict ({outcome.error or outcome.result[:200]!r})")
        scores = {c: int(min(10, max(1, round(float(out[c]))))) for c in CRITERIA}
        return Verdict(scores=scores, critique=str(out.get("critique", "")).strip()[:1500],
                       session_id=outcome.session_id, cost=outcome.cost)


def _json_in(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

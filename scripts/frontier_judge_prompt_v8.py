"""Complete citation chains under the unchanged documented-support rubric.

Developed after observing v7's 192 synthetic controls. Those controls are
development regression, not blind validation or scientific gold. The old prompt
and all old predictions remain frozen; this profile generates new predictions.
"""
from frontier_judge_prompt_v5 import SYSTEM as PRECEDENCE_SYSTEM

VERSION = "pbpk-documented-support-complete-citations-v8"

CITATION_COMPLETENESS = """
Choose a COMPLETE, traceable citation set for the decision, not just the excerpt
asserting the final applicability conclusion. When an excerpt relies on a method,
equation, workflow or result defined in a different supplied excerpt (for example,
it refers to the preceding specification), cite that defining excerpt AND the
excerpt establishing the relevant transfer, adaptation or current-version link.
A bridge that only asserts applicability or repeats task/endpoint names does not
itself provide the referenced method or structure. If its own substantive prose
also provides that complete definition, one excerpt is sufficient.
Also include the excerpts needed to establish the explicitly requested scope.
One excerpt is enough when its own substantive prose establishes the entire
decision without such a dependency; do not add irrelevant or redundant citations.
Use at most three existing excerpt numbers. If a complete support chain cannot
be established within the supplied evidence and citation limit, do not assert
the unsupported conclusion. Never invent a link, method detail or evidence ID.
"""

_ANCHOR = "\nFinal check: unknown required information"
assert PRECEDENCE_SYSTEM.count(_ANCHOR) == 1
SYSTEM = PRECEDENCE_SYSTEM.replace(_ANCHOR, "\n" + CITATION_COMPLETENESS + _ANCHOR)

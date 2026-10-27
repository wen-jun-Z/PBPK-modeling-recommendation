"""Evidence-first serialization of the unchanged v2 documented-support rubric.

The v2 production run stopped on unsupported, uncited decisions. This version
changes the order in which a judgment is emitted, not the relevance scale or
the evidence requirements. Freeze this module before new validation outputs.
"""

from frontier_judge_prompt_v2 import SYSTEM as V2_SYSTEM

VERSION = "pbpk-documented-support-evidence-first-v3"

_OLD_OUTPUT = '''Return ONLY a JSON array, with exactly one object per request and no explanation:
[{"i":1,"g":null,"a":"U","e":[]}]. Use fields i,g,a,e only.
'''

if V2_SYSTEM.count(_OLD_OUTPUT) != 1:
    raise ValueError("The frozen v2 output instruction changed")

SYSTEM = V2_SYSTEM.replace(_OLD_OUTPUT, '''Return ONLY a JSON array, with exactly one object per request and no explanation.
Write each object's fields in this order: i, e, g, a. Select the evidence BEFORE
emitting the grade and applicability. Use only excerpt numbers that support the
decision under the rules above; a valid number alone does not establish support.
Then assign g and a based on that selected evidence and the requested main use.
If no excerpt supports a decision, choose e=[] and report g=null,a="U". A cited
statement of missing information may also support g=null,a="U". Never choose an
irrelevant citation merely to make a definite judgment possible.
The output shape for an unsupported decision is:
[{"i":1,"e":[],"g":null,"a":"U"}]. Use fields i,e,g,a only.
''')

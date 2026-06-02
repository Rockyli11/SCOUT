"""D1 rule-based prompt-injection detector."""

from __future__ import annotations

import re

from scout_router.detectors.base import DetectorBase
from scout_router.detectors.registry import register_detector
from scout_router.schema import DetectorResult, PromptSample

OVERRIDE_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"forget\s+(everything|all|your\s+instructions?)",
    r"do\s+not\s+follow\s+(your\s+)?(previous|prior|original)\s+instructions?",
    r"override\s+(your\s+)?(instructions?|system\s+prompt|rules?)",
    r"new\s+instructions?\s*:",
    r"your\s+(new\s+)?instructions?\s+(are|is)\s*:",
    r"from\s+now\s+on[,\s]",
    r"instead[,\s]+do\s+the\s+following",
    r"stop\s+(what\s+you('re|\s+are)\s+doing|following)",
]
EXFILTRATION_PATTERNS = [
    r"(print|output|reveal|expose|show|leak|dump|repeat|return|send)\s+(the\s+)?(system\s+prompt|instructions?|api\s+key|password|secret|credential|token|private)",
    r"what\s+(are|is)\s+your\s+(system\s+prompt|instructions?|rules?)",
    r"repeat\s+(everything|all|your\s+instructions?)",
]
ROLE_CHANGE_PATTERNS = [
    r"you\s+are\s+now\s+(a\s+|an\s+)?(?!assistant|helpful)",
    r"act\s+as\s+(a\s+|an\s+)?\w+\s+without\s+(any\s+)?(restriction|filter|limit|constraint)",
    r"pretend\s+(you\s+are|to\s+be)\s+(a\s+|an\s+)?(?!assistant)",
    r"(jailbreak|dan|do\s+anything\s+now)",
    r"developer\s+mode",
]
TOOL_PATTERNS = [
    r"call\s+(the\s+)?\w+\s+tool\s+(with|using|and)",
    r"use\s+(the\s+)?\w+\s+function\s+to",
    r"execute\s+(the\s+following|this)\s+command",
    r"run\s+the\s+following\s+(command|script|code)",
    r"send\s+(this|the\s+following)\s+(to|email|message)\s+",
]
ENCODING_PATTERNS = [
    r"base64[_\s]*(decode|encoded)",
    r"(?:[A-Za-z0-9+/]{20,}={0,2})",
    r"\\u[0-9a-fA-F]{4}",
    r"\u200b|\u200c|\u200d|\ufeff",
]

ALL_PATTERNS = OVERRIDE_PATTERNS + EXFILTRATION_PATTERNS + ROLE_CHANGE_PATTERNS + TOOL_PATTERNS + ENCODING_PATTERNS
_COMPILED = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in ALL_PATTERNS]
_GROUP_WEIGHTS = (
    [0.9] * len(OVERRIDE_PATTERNS)
    + [0.85] * len(EXFILTRATION_PATTERNS)
    + [0.8] * len(ROLE_CHANGE_PATTERNS)
    + [0.75] * len(TOOL_PATTERNS)
    + [0.6] * len(ENCODING_PATTERNS)
)


@register_detector("d1_rule_based")
class RuleBasedDetector(DetectorBase):
    name = "d1_rule_based"
    cost_tier = "cheap"

    def _detect(self, sample: PromptSample) -> DetectorResult:
        hits = []
        for pattern, weight, compiled in zip(ALL_PATTERNS, _GROUP_WEIGHTS, _COMPILED):
            if compiled.search(sample.eval_content):
                hits.append((pattern, weight))

        if not hits:
            return DetectorResult(label=0, confidence=0.05, raw={"hits": []})

        miss_prob = 1.0
        for _, weight in hits:
            miss_prob *= 1.0 - weight
        confidence = min(1.0 - miss_prob, 0.99)
        return DetectorResult(label=1, confidence=confidence, raw={"hits": [pattern for pattern, _ in hits]})

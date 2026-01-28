from __future__ import annotations
import re
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class PlayTypeRules:
    rules: Dict[str, List[re.Pattern]]

    @classmethod
    def load(cls, path: str) -> "PlayTypeRules":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        rules = {}
        for k, patterns in raw.items():
            rules[k] = [re.compile(p) for p in patterns]
        return cls(rules)

    def classify(self, play_type: str) -> Optional[str]:
        if not play_type:
            return None
        for cat, pats in self.rules.items():
            for p in pats:
                if p.search(play_type):
                    return cat
        return None

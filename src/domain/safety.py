"""Rule-based input safety classifier for the RAG pipeline.

Evaluates the ordered ``risk_rules`` from safety_config.json (first match
wins); adding a rule or keywords needs no code changes. Extracted from
NLPController so orchestration, scoring, and verification stay separate.
"""
import re

from helpers.safety_config import compile_rule_pattern, config_signature, get_safety_config
from models.enums.SafetyEnums import RiskLevel, SafetyReason


class SafetyClassifier:

    _ANIMAL_TERMS = ("dog", "cat", "pet", "animal", "horse", "bird")

    _compiled_cache = {}

    def _is_personal_symptom_query(self, query: str) -> bool:
        """Rule-based guardrail: detect when the user asks about the symptoms of
        a specific person (self or family) so the answer can refuse to diagnose.
        Patterns are sourced from safety_config.json (patterns.*)."""
        text = query or ""
        patterns = (get_safety_config().get("patterns") or {})

        person_ref = self._compiled_named_pattern("person_ref", patterns)
        first_person = self._compiled_named_pattern("first_person", patterns)
        strong_symptom = self._compiled_named_pattern("strong_symptom", patterns)
        health_topic = self._compiled_named_pattern("health_topic", patterns)

        has_person = bool(person_ref.search(text))
        has_first = bool(first_person.search(text))
        has_strong = bool(strong_symptom.search(text))
        has_topic = bool(health_topic.search(text))

        return (has_person and has_topic) or (has_first and has_strong)

    def _is_animal_patient_query(self, query: str) -> bool:
        """True when the query asks about an animal's own health (animal as
        patient), e.g. 'My cat is wheezing'.  Returns False for questions
        where the animal is an allergen/trigger context, e.g. 'Can pet
        dander trigger asthma?' — those are in-scope clinical questions."""
        text = (query or "").lower()
        has_animal = any(
            re.search(r"\b" + term + r"\b", text) for term in self._ANIMAL_TERMS
        )
        if not has_animal:
            return False
        possessives = ("my ", "our ", "his ", "her ")
        return any(p + a in text for p in possessives for a in self._ANIMAL_TERMS)

    def _compiled_named_pattern(self, name: str, patterns: dict):
        entry = patterns.get(name)
        if not entry:
            return re.compile(r"(?!)")
        cache_key = (name, self._config_signature())
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]
        compiled = compile_rule_pattern(entry.get("patterns"))
        self._compiled_cache[cache_key] = compiled
        return compiled

    @staticmethod
    def _config_signature():
        return config_signature()

    def _compiled_rule_pattern(self, name: str, rule: dict):
        patterns = rule.get("patterns")
        if not patterns:
            return None
        cache_key = (name, self._config_signature())
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]
        compiled = compile_rule_pattern(patterns)
        self._compiled_cache[cache_key] = compiled
        return compiled

    def _rule_matches(self, rule: dict, lowered: str, text: str) -> bool:
        pattern = self._compiled_rule_pattern(rule.get("name"), rule)
        if pattern is not None and not pattern.search(lowered):
            return False
        patterns = (get_safety_config().get("patterns") or {})
        for veto_name in rule.get("when_not") or []:
            if self._compiled_named_pattern(veto_name, patterns).search(lowered):
                return False
        for require_name in rule.get("require") or []:
            if not self._compiled_named_pattern(require_name, patterns).search(lowered):
                return False
        if rule.get("auxiliary"):
            handler = getattr(self, rule["auxiliary"], None)
            if handler is None or not handler(text):
                return False
        return True

    def classify_input_risk(self, query: str):
        text = (query or "").strip()
        lowered = text.lower()

        if not text:
            return self._risk(RiskLevel.REFUSE_REDIRECT.value, SafetyReason.EMPTY_QUERY.value)

        # Rules are evaluated in order; the first match wins. Adding a new rule
        # (or keywords/patterns to an existing one) is done in
        # src/config/safety_config.json, no code changes required.
        config = get_safety_config()
        for rule in config.get("risk_rules") or []:
            if self._rule_matches(rule, lowered, text):
                return self._risk(rule.get("risk_level", RiskLevel.ALLOWED.value), rule.get("reason"))

        return self._risk(RiskLevel.ALLOWED.value, SafetyReason.WITHIN_ASTHMA_GUIDELINE_SCOPE.value)

    @staticmethod
    def _risk(risk_level: str, reason: str):
        return {
            "risk_level": risk_level,
            "reason": reason,
        }

    def _build_refusal_answer(self, risk_assessment: dict):
        config = get_safety_config()
        refusals = config.get("refusals") or {}
        default = config.get("refusal_default") or (
            "I can only answer clinical questions that are within the indexed asthma "
            "guideline scope and supported by retrieved evidence."
        )
        return refusals.get(risk_assessment.get("reason"), default)

    @staticmethod
    def _clinical_disclaimer():
        return (
            "This system supports clinical guideline review and does not replace "
            "clinical judgment, diagnosis, emergency care, or advice from a qualified "
            "healthcare professional."
        )

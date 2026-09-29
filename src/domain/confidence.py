"""Pre/post-generation confidence scoring for the RAG pipeline.

`build_confidence` is a pure pre-generation gate signal (retrieval score +
official-evidence coverage + dosing guardrail). `apply_post_generation`
folds citation/claim verification into a display copy without touching the
gate. Extracted from NLPController.
"""
from domain import text as text_utils
from helpers.config import get_settings


class ConfidenceScorer:

    _CONFIDENCE_DOWNGRADE = {"high": "medium", "medium": "low", "low": "low"}

    def _build_confidence(self, retrieved_documents: list, selected_documents: list, question: str = "", settings=None):
        # `settings` is injectable so callers (and tests) can supply
        # configuration without patching this module's import.
        settings = settings or get_settings()
        scores = [
            doc.get("score")
            for doc in retrieved_documents or []
            if isinstance(doc.get("score"), (int, float))
        ]
        top_score = max(scores) if scores else None
        evidence_count = sum(
            1 for doc in selected_documents or []
            if text_utils.has_official_evidence_metadata(doc)
        )
        generation_allowed = evidence_count >= settings.ANSWER_MIN_EVIDENCE_COUNT

        if (
            settings.ANSWER_MIN_TOP_SCORE > 0
            and top_score is not None
            and top_score < settings.ANSWER_MIN_TOP_SCORE
        ):
            generation_allowed = False

        # Dosing-query guardrail: numeric, unit, or frequency questions are
        # blocked unless the retrieved evidence carries numeric content, so we
        # never answer with a made-up dose.
        if generation_allowed and text_utils.is_numeric_or_dosing_query(question):
            evidence_numeric = any(
                text_utils.numeric_values(" ".join(
                    str(part) for part in [
                        doc.get("text"),
                        (doc.get("metadata") or {}).get("excerpt") or (doc.get("metadata") or {}).get("snippet") or "",
                    ] if part
                ))
                for doc in selected_documents or []
            )
            if not evidence_numeric:
                generation_allowed = False

        if not generation_allowed:
            level = "insufficient"
            reason = "insufficient_official_evidence"
        elif top_score is None:
            level = "medium" if evidence_count >= 2 else "low"
            reason = "no_numeric_score"
        elif top_score >= 0.65:
            level = "high"
            reason = "strong_retrieval_score"
        elif top_score >= 0.30:
            level = "medium"
            reason = "moderate_retrieval_score"
        else:
            level = "low"
            reason = "weak_retrieval_score"

        return {
            "confidence_level": level,
            "top_score": top_score,
            "evidence_count": evidence_count,
            "generation_allowed": generation_allowed,
            "reason": reason,
        }

    @classmethod
    def _apply_post_generation_confidence(cls, confidence: dict, quality: dict):
        """Slide 11 ('Add a confidence level') requires confidence to reflect
        citation coverage and safety checks, not just the pre-generation
        retrieval score. `_build_confidence` runs before generation (it gates
        whether generation happens at all) and must stay a pure pre-generation
        signal for its existing tests. This folds the post-generation
        citation/claim verification outcome into a copy shown to the caller,
        without touching the gate itself."""
        level = confidence.get("confidence_level")
        if level not in cls._CONFIDENCE_DOWNGRADE:
            return confidence

        failed_verification = (
            quality.get("citation_faithfulness", 1.0) < 1.0
            or quality.get("unsupported_claim_rate", 0.0) > 0.0
        )
        if not failed_verification:
            return confidence

        downgraded = dict(confidence)
        downgraded["confidence_level"] = cls._CONFIDENCE_DOWNGRADE[level]
        downgraded["downgrade_reason"] = "citation_or_claim_verification_failed"
        return downgraded

    @staticmethod
    def _build_confidence_summary(confidence: dict):
        if not confidence.get("generation_allowed"):
            return "Insufficient — generation blocked due to insufficient official evidence"

        level = confidence.get("confidence_level", "low").capitalize()
        score = confidence.get("top_score")
        score_str = f"{score:.2f}" if score is not None else "N/A"
        reason = confidence.get("reason", "").replace("_", " ")
        evidence_count = confidence.get("evidence_count", 0)

        return f"{level} — {reason} ({score_str}), {evidence_count} official evidence chunks"

"""Post-retrieval answer verification: citations, evidence, claim support.

Pure logic over retrieved documents and generated text — no LLM, vector,
or DB clients. Extracted from NLPController; the controller keeps thin
delegates so routes, the answer pipeline, and existing tests are untouched.
"""
import re

from domain import text as text_utils
from helpers.safety_config import get_safety_config


class AnswerQualityEvaluator:

    ORG_ALIASES = None

    CITATION_STOP_TOKENS = {
        "the", "a", "an", "and", "or", "of", "for", "in", "on", "to",
        "with", "at", "by", "is", "are", "was", "be", "as", "pp", "p",
    }

    @classmethod
    def _get_org_aliases(cls):
        if cls.ORG_ALIASES is None:
            cls.ORG_ALIASES = (get_safety_config().get("org_aliases") or {})
        return cls.ORG_ALIASES

    @classmethod
    def _doc_tokens(cls, text: str):
        return {
            token
            for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if token not in cls.CITATION_STOP_TOKENS
        }

    @classmethod
    def _citation_doc_score(cls, document: str, source_doc: str, source_url: str):
        """0.0 = no match, >=0.6 = fuzzy token-overlap match, 1.0 = exact match."""
        normalized_doc = document.lower()
        normalized_source = source_doc.lower()
        if normalized_doc in normalized_source or normalized_source in normalized_doc:
            return 1.0

        citation_tokens = cls._doc_tokens(document)
        source_tokens = cls._doc_tokens(source_doc)
        if not citation_tokens:
            return 0.0

        haystack = f"{normalized_source} {(source_url or '').lower()}"
        for token in citation_tokens:
            for org, markers in cls._get_org_aliases().items():
                if token == org and any(marker in haystack for marker in markers):
                    return 1.0

        overlap = len(citation_tokens & source_tokens)
        if overlap == 0:
            return 0.0
        return overlap / min(len(citation_tokens), len(source_tokens))

    @classmethod
    def _sections_match(cls, cited_section: str, source_section: str, source_text: str = ""):
        cited = text_utils.normalize_section(cited_section)
        source = text_utils.normalize_section(source_section)
        if not cited:
            return True
        if cited in source or source in cited:
            return True
        # Numbered refs like "1.2.1" rarely appear in the extracted PDF
        # section_title; accept only when the chunk body carries the parent
        # section marker (e.g. "1.2") that the cited number descends from.
        match = re.match(r"(\d+(?:\.\d+)+)", cited)
        if not match:
            return False
        number = match.group(1)
        if number in (source_text or ""):
            return True
        parent = number.rsplit(".", 1)[0]
        return bool(parent and parent in (source_text or ""))

    @classmethod
    def _find_cited_source(cls, document: str, page_number: int, section: str, sources: list):
        best_match = None
        best_score = 0.0

        for source in sources or []:
            source_doc = (source.get("document_name") or source.get("file_name") or "")
            source_section = (source.get("section_title") or "")
            source_page = source.get("page_number")
            try:
                source_page = int(source_page)
            except (TypeError, ValueError):
                continue

            page_matches = source_page == page_number
            section_matches = cls._sections_match(
                section, source_section, source.get("text") or ""
            )

            if not (page_matches and section_matches):
                continue

            score = cls._citation_doc_score(
                document, source_doc, source.get("source_url") or ""
            )
            if score > best_score:
                best_score = score
                best_match = source

        if best_score >= 0.6:
            return best_match
        return None

    def _build_sources(self, selected_documents: list):
        sources = []

        for idx, doc in enumerate(selected_documents):
            metadata = doc.get("metadata") or {}
            text = doc.get("text") or ""
            sources.append({
                "doc_num": idx + 1,
                "chunk_id": doc.get("chunk_id"),
                "file_name": metadata.get("file_name") or metadata.get("file_id"),
                "document_name": metadata.get("document_name") or metadata.get("file_name"),
                "asset_id": metadata.get("asset_id"),
                "chunk_order": metadata.get("chunk_order"),
                "page_number": metadata.get("page_number"),
                "section_title": metadata.get("section_title"),
                "source_url": metadata.get("source_url"),
                "score": doc.get("score"),
                "text": text[:300]
            })

        return sources

    def build_evidence_panel(self, retrieved_documents: list, selected_documents: list):
        retrieved_count = len(retrieved_documents) if retrieved_documents else 0
        selected_count = len(selected_documents) if selected_documents else 0

        unique_docs = set()
        pages = []
        chunks = []

        for idx, doc in enumerate(selected_documents or []):
            metadata = doc.get("metadata") or {}
            doc_name = metadata.get("document_name") or metadata.get("file_name") or "Unknown"
            page_num = metadata.get("page_number")

            unique_docs.add(doc_name)
            if page_num is not None:
                try:
                    pages.append(int(page_num))
                except ValueError:
                    pass

            chunks.append({
                "rank": idx + 1,
                "chunk_id": doc.get("chunk_id"),
                "relevance_score": doc.get("score"),
                "document_name": doc_name,
                "page_number": page_num,
                "section_title": metadata.get("section_title") or "",
                "source_url": metadata.get("source_url") or "",
                "text_preview": (doc.get("text") or "")[:300],
                "is_official_source": text_utils.has_official_evidence_metadata(doc)
            })

        page_range = {}
        if pages:
            page_range = {"min": min(pages), "max": max(pages)}

        return {
            "total_retrieved": retrieved_count,
            "total_selected": selected_count,
            "retrieval_coverage": {
                "documents": list(unique_docs),
                "unique_documents": len(unique_docs),
                "page_range": page_range
            },
            "chunks": chunks
        }

    def verify_citations(self, answer: str, sources: list):
        citations = []
        citation_pattern = re.compile(
            r"[\[\(](?P<document>[^\[\]()]+?)[,:]\s*(?:pp?\.?|pages?|pg\.?)\s*"
            r"(?P<page>\d+)(?:\s*[-,]\s*\d+)?(?:,\s*(?P<section>[^\[\]]+?))?[\]\)]",
            re.IGNORECASE,
        )

        # Pre-process: split combined citations like [Doc1, p. 1, Doc2, p. 2]
        # into separate [Doc1, p. 1] and [Doc2, p. 2] brackets.
        def _split_combined_citations(text):
            def _split_match(m):
                inner = m.group(0)[1:-1]  # strip [ ]
                # Split on document boundaries: look for known doc names followed by page refs
                parts = re.split(r'(?=(?:GINA|NICE|NHLBI|WHO|CDC|USPSTF)(?:[_\s]|$))', inner)
                parts = [p.strip().strip(',').strip() for p in parts if p.strip().strip(',').strip()]
                if len(parts) <= 1:
                    return m.group(0)
                return ']['.join(f'[{p}]' for p in parts)
            return re.sub(r'\[[^\[\]]{20,}\]', _split_match, text)

        processed_answer = _split_combined_citations(answer or "")

        for match in citation_pattern.finditer(processed_answer):
            document = match.group("document").strip()
            page = int(match.group("page"))
            section = (match.group("section") or "").strip()
            matched_source = self._find_cited_source(document, page, section, sources)
            citations.append({
                "citation": match.group(0),
                "document": document,
                "page_number": page,
                "section_title": section,
                "supported": matched_source is not None,
                "source_chunk_id": matched_source.get("chunk_id") if matched_source else None,
            })

        return citations

    def build_answer_quality(self, answer: str, sources: list, selected_documents: list,
                             verify_claims: bool = True):
        citations = self.verify_citations(answer, sources)
        citation_total = len(citations)
        supported_total = sum(1 for citation in citations if citation.get("supported"))
        citation_faithfulness = (
            supported_total / citation_total
            if citation_total > 0
            else 1.0 if (not answer or not verify_claims) else 0.0
        )
        unsupported_claims = (
            self.detect_unsupported_claims(answer, selected_documents, citations)
            if verify_claims
            else []
        )
        claims_checked = len(self._extract_claims(answer)) if verify_claims else 0
        unsupported_claim_rate = (
            len(unsupported_claims) / claims_checked
            if claims_checked > 0
            else 0.0
        )

        return {
            "citations": citations,
            "citation_faithfulness": round(citation_faithfulness, 3),
            "unsupported_claims": unsupported_claims,
            "unsupported_claim_rate": round(unsupported_claim_rate, 3),
            "claims_checked": claims_checked,
        }

    def detect_unsupported_claims(self, answer: str, selected_documents: list, citations: list):
        unsupported = []
        cited_chunk_ids = {
            citation.get("source_chunk_id")
            for citation in citations
            if citation.get("supported") and citation.get("source_chunk_id") is not None
        }

        sources_cache = self._build_sources(selected_documents)
        for claim in self._extract_claims(answer):
            claim_citations = self.verify_citations(claim, sources_cache)
            supported_citations = [citation for citation in claim_citations if citation.get("supported")]
            evidence_text = self._evidence_text_for_claim(selected_documents, supported_citations, cited_chunk_ids)
            support_score = self._claim_support_score(claim, evidence_text)
            # Arabic claims against English evidence cannot be compared via
            # token overlap. Skip the deterministic verdict for those; their
            # support is carried by citation faithfulness (the citations still
            # carry the exact English document names and page numbers).
            if text_utils.contains_arabic(claim) and not text_utils.contains_arabic(evidence_text):
                continue
            if support_score < 0.15:
                unsupported.append({
                    "claim": claim,
                    "reason": "weak_evidence_overlap" if claim_citations else "uncited_and_weak_evidence_overlap",
                    "support_score": round(support_score, 3),
                })

        return unsupported

    def _extract_claims(self, answer: str):
        text = (answer or "").strip()
        if not text:
            return []

        skip_patterns = re.compile(
            r"\b(i can'?t diagnose|consult a qualified|emergency medical|"
            r"local emergency services|here is general information|i do not have enough)\b",
            re.IGNORECASE,
        )

        # Mask citation brackets so the period in "p. 12]" never splits a claim,
        # and keep line breaks so each bullet stays its own claim.
        def _mask_line(snippet):
            citations = []

            def _replace(match):
                citations.append(match.group(0))
                return f"\uE000CIT{len(citations)}\uE001"
            return re.sub(r"\[[^\[\]]+\]", _replace, snippet), citations

        def _unmask(fragment, citations):
            def _restore(match):
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(citations):
                    return citations[idx]
                return match.group(0)
            return re.sub(r"\uE000CIT(\d+)\uE001", _restore, fragment)

        claims = []
        for raw_line in re.split(r"\n+", text):
            line = raw_line.strip()
            line = re.sub(r"^[-*•]\s+", "", line)
            if not line:
                continue
            masked_line, citations = _mask_line(line)
            for fragment in re.split(r"(?<=[.!?؟])\s+", masked_line):
                claim = _unmask(fragment.strip(), citations)
                if len(claim.split()) < 5:
                    continue
                if skip_patterns.search(claim):
                    continue
                claims.append(claim)

        return claims

    def _evidence_text_for_claim(self, selected_documents: list, supported_citations: list,
                                 fallback_chunk_ids: set):
        chunk_ids = {
            citation.get("source_chunk_id")
            for citation in supported_citations
            if citation.get("source_chunk_id") is not None
        }
        if not chunk_ids:
            chunk_ids = fallback_chunk_ids

        texts = []
        for doc in selected_documents or []:
            if chunk_ids and doc.get("chunk_id") not in chunk_ids:
                continue
            texts.append(doc.get("text") or "")

        if not texts:
            texts = [doc.get("text") or "" for doc in selected_documents or []]

        return " ".join(texts)

    def _claim_support_score(self, claim: str, evidence_text: str):
        claim_tokens = text_utils.meaningful_tokens(claim)
        if not claim_tokens:
            return 0.0

        evidence_tokens = set(text_utils.meaningful_tokens(evidence_text))
        if not evidence_tokens:
            return 0.0

        overlap = sum(1 for token in claim_tokens if token in evidence_tokens)
        token_score = overlap / len(claim_tokens)

        # Numeric/unit agreement: a claim carrying figures (doses, ranges,
        # frequencies, percentages) is only strongly supported when the
        # evidence carries the same values.
        claim_values = text_utils.numeric_values(claim)
        if claim_values:
            evidence_values = text_utils.numeric_values(evidence_text)
            matched_values = sum(1 for value in claim_values if value in evidence_values)
            numeric_ratio = matched_values / len(claim_values)
            return 0.55 * token_score + 0.45 * numeric_ratio

        return token_score

from typing import List
import re, asyncio

from models.db_schemas import DataChunk
from stores.llm.LLMEnums import DocumentType
from helpers.config import get_settings
from helpers.safety_config import get_safety_config

from .BaseController import BaseController
from domain.safety import SafetyClassifier
from domain.quality import AnswerQualityEvaluator
from domain.confidence import ConfidenceScorer
from models.db_schemas import Project
import json
DOC_HEADER_OVERHEAD_CHARS = 220
FOOTER_RESERVE_CHARS = 900 
class NLPController(BaseController):
    
    
    
    def __init__(self, vectordb_client,generation_client,embedding_client,template_parser, rerank_client=None):
        super().__init__()
        self.vectordb_client=vectordb_client
        self.generation_client=generation_client
        self.embedding_client=embedding_client
        self.template_parser=template_parser
        self.rerank_client=rerank_client
        self.safety = SafetyClassifier()
        self.evaluator = AnswerQualityEvaluator()
        self.scorer = ConfidenceScorer()
        
    _EXPAND_PATTERNS = [
        (re.compile(r"\bmart\b|maintenance[-\s]?and[-\s]?reliever", re.I), "maintenance-and-reliever therapy ICS-formoterol budesonide formoterol as-needed single-inhaler combination"),
        (re.compile(r"\bair\b|anti[-\s]?inflammatory reliever", re.I), "anti-inflammatory reliever low dose ICS-formoterol as-needed"),
        (re.compile(r"\bics\b|inhaled corticosteroid", re.I), "inhaled corticosteroid controller preventer corticosteroid"),
        (re.compile(r"\bsaba\b|short[-\s]?acting beta", re.I), "short-acting beta agonist reliever salbutamol albuterol SABA-only"),
        (re.compile(r"\blaba\b|long[-\s]?acting beta", re.I), "long-acting beta agonist formoterol salmeterol"),
        (re.compile(r"\baction plan\b|self[-\s]?management", re.I), "written asthma action plan personalised action plan monitor symptoms seek medical care"),
        (re.compile(r"\bobesity\b|overweight|weight", re.I), "weight reduction 5-10% 5–10% improve asthma control obesity comorbidity"),
        (re.compile(r"\binitial\b|newly diagnosed|first treatment", re.I), "initial pharmacological treatment low-dose ICS/formoterol as-needed AIR therapy"),
        (re.compile(r"\bstep 1\b|step one", re.I), "Step 1 preferred treatment adults adolescents low dose ICS-formoterol"),
        (re.compile(r"\bdiagnos", re.I), "confirm asthma diagnosis spirometry bronchodilator reversibility FeNO eosinophil variability"),
        (re.compile(r"\bexercise\b|bronchospasm", re.I), "exercise-induced bronchospasm before exercise SABA LTRA cromolyn NHLBI"),
        (re.compile(r"\bgina\b", re.I), "GINA 2026 Global Strategy for Asthma Management and Prevention GINA 2026 Summary Guide for Asthma Management and Prevention"),
        (re.compile(r"\bnice\b|ng80|bts", re.I), "NICE Asthma: diagnosis monitoring and chronic asthma management NG80"),
        (re.compile(r"\bnhlbi\b|naepp|epr4|epr[-\s]?4", re.I), "Asthma Quick Reference Guide 2020 NAEPP EPR-4 focused updates NHLBI"),
        (re.compile(r"step[-\s]?down|stepped?\s+down|de-?escalat|reduc\w*\s+(treatment|therapy|dose)", re.I),
        "stepping down maintenance therapy reduce controller dose de-escalation well-controlled asthma minimum effective treatment"),
        (re.compile(r"non[-\s]?pharmacolog|smoking cessation|vaccination|immunisation|obesity|weight reduction", re.I),
        "non-pharmacological strategies smoking cessation physical activity weight reduction vaccination obesity self-management GINA guideline recommendations"),
        (re.compile(r"risk.*(exacerb|poor outcome|severe)|increased risk|identify.*risk|at risk", re.I),
        "risk factors exacerbations severe exacerbation poor lung function FEV1 intubation past year emergency visit"),
        (re.compile(r"preferred treatment|step.?1.*persistent|persistent.*step.?1", re.I),
        "preferred treatment Step 1 persistent asthma low-dose ICS SABA as needed NHLBI stepwise approach"),
    ]
    def create_collection_name(self,project_id:int):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()
    
    async def reset_vector_db_collection(self,project:Project):
        collection_name=self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name)
    
    async def get_vector_db_collection_info(self ,project:Project):
        collection_name=self.create_collection_name(project_id=project.project_id)
        collection_info=await self.vectordb_client.get_collection_info(collection_name)
        
        return json.loads(
            json.dumps(collection_info,default=lambda x: x.__dict__)
        )
    
    async def index_into_vector_db(self,project:Project,
                             chunks:List[DataChunk],
                             chunk_ids:List[int],
                             do_reset:bool=False
                             ):
        collection_name=self.create_collection_name(project_id=project.project_id)
        filtered_chunks=[c for c in chunks if c.chunk_text and len(c.chunk_text.strip()) > 1]
        texts=[c.chunk_text.strip() for c in filtered_chunks]
        metadata=[self._parse_chunk_metadata(c.chunk_metadata) for c in filtered_chunks]
        record_ids=[c.chunk_id for c in filtered_chunks]
        settings=get_settings()

        if not texts:
            return True

        vectors=[]
        for i in range(0, len(texts), settings.EMBEDDING_BATCH_SIZE):
            batch_vectors=await self.embedding_client.embed_text(
                text=texts[i:i + settings.EMBEDDING_BATCH_SIZE],
                document_type=DocumentType.DOCUMENT.value
            )
            if not batch_vectors:
                return False
            vectors.extend(batch_vectors)

        if len(vectors) != len(texts):
            return False

        
        _= await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
            
            
        )
        _= await self.vectordb_client.insert_many(collection_name=collection_name,
                                            texts=texts,
                                            metadata= metadata,
                                            vector=vectors,
                                            record_ids=record_ids,
                                            batch_size=settings.VECTOR_INSERT_BATCH_SIZE
                                            )
        
        
        return True
    
    
    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 10,
                                          score_threshold: float = None,
                                          metadata_filter: dict = None,
                                          retrieval_mode: str = "hybrid",
                                          rerank: bool = False,
                                          rerank_top_k: int = None,
                                          expand_query: bool = True):
        """
        Retrieve chunks from the project collection.

        retrieval_mode: "vector" | "keyword" | "hybrid"
        rerank: if True, fetch retrieval_mode candidates (top RETRIEVAL_TOP_K)
                then reorder with the rerank_client and trim to `limit`.
        Returns a JSON-safe list of dicts.
        """
        query_vector = None
        collection_name = self.create_collection_name(project_id=project.project_id)
        expanded_query = self.expand_query(text) if expand_query else text
        raw_query = text

        if retrieval_mode == "keyword":
            results = await self.vectordb_client.search_by_keyword(
                collection_name=collection_name,
                query=raw_query,
                limit=limit,
                metadata_filter=metadata_filter,
            )
        elif retrieval_mode == "vector":
            vectors = await self.embedding_client.embed_text(
                text=expanded_query,
                document_type=DocumentType.QUERY.value
            )
            if vectors and isinstance(vectors, list) and len(vectors) > 0:
                query_vector = vectors[0]
            if query_vector is None:
                return None
            results = await self.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                metadata_filter=metadata_filter,
            )
        else:  # hybrid
            embed_task = asyncio.ensure_future(self.embedding_client.embed_text(
                text=expanded_query,
                document_type=DocumentType.QUERY.value
            ))
            keyword_task = asyncio.ensure_future(self.vectordb_client.search_by_keyword(
                collection_name=collection_name,
                query=raw_query,
                limit=limit,
                metadata_filter=metadata_filter,
            ))
            vectors, keyword_results = await asyncio.gather(embed_task, keyword_task)

            if vectors and isinstance(vectors, list) and len(vectors) > 0:
                query_vector = vectors[0]
            if query_vector is None:
                return None

            settings = get_settings()
            results = await self.vectordb_client.search_hybrid(
                collection_name=collection_name,
                vector=query_vector,
                query=raw_query,
                limit=limit,
                metadata_filter=metadata_filter,
                rrf_k=settings.HYBRID_RRF_K,
            )

        if not results:
            return [] if results == [] else None

        if rerank and self.rerank_client and query_vector is not None:
            settings = get_settings()
            candidate_k = max(limit, settings.RETRIEVAL_TOP_K)
            rerank_limit = rerank_top_k or settings.RERANK_TOP_K

            candidates = await self._build_rerank_candidates(
                collection_name=collection_name,
                query=raw_query,
                vector=query_vector,
                limit=candidate_k,
                metadata_filter=metadata_filter,
                score_threshold=score_threshold,
                fallback_results=results,
            )

            reranked = await self.rerank_client.rerank(
                query=expanded_query,
                documents=candidates or results,
                top_n=rerank_limit,
            )
            results = reranked[:limit]

        return json.loads(
            json.dumps(results, default=lambda x: x.__dict__)
        )
        
        
    async def answer_rag_question(self,project:Project,query:str,limit:int =5,
                                  score_threshold:float=None,
                                  metadata_filter:dict=None,
                                  include_sources:bool=True,
                                  retrieval_mode:str="hybrid",
                                  rerank:bool=True,
                                  rerank_top_k:int=None,
                                  expand_query:bool=True,
                                  verify_claims:bool=True,
                                  conversation_history:list=None):

        answer , full_prompt , chat_history=None,None,None
        settings=get_settings()
        conversation_block=self._build_conversation_block(conversation_history)
        retrieval_query=self._build_retrieval_query(query, conversation_history)
        expanded_query = self.expand_query(retrieval_query) if expand_query else retrieval_query
        risk_assessment=self.classify_input_risk(query)
        refusal_answer=self._build_refusal_answer(risk_assessment)
        disclaimer=self._clinical_disclaimer()

        if risk_assessment["risk_level"] == "refuse_redirect":
            confidence=self._build_confidence([], [], question=query)
            confidence["generation_allowed"]=False
            confidence["reason"]="blocked_by_safety_classifier"
            quality=self.build_answer_quality(refusal_answer, [], [], verify_claims=False)
            evidence_panel = {"total_retrieved": 0, "total_selected": 0, "retrieval_coverage": {"documents": [], "unique_documents": 0, "page_range": {}}, "chunks": []}
            return refusal_answer, None, None, [], risk_assessment, confidence, quality, disclaimer, evidence_panel, expanded_query
        
        retrived_document= await self.search_vector_db_collection(
            project=project,
            text=retrieval_query,
            limit=max(limit, settings.RETRIEVAL_TOP_K),
            score_threshold=score_threshold if score_threshold is not None else settings.RETRIEVAL_SCORE_THRESHOLD,
            metadata_filter=metadata_filter,
            retrieval_mode=retrieval_mode,
            rerank=rerank,
            rerank_top_k=rerank_top_k or settings.RERANK_TOP_K,
            expand_query=expand_query,
        )
        
        if not retrived_document or len(retrived_document)==0:
            confidence=self._build_confidence([], [], question=query)
            quality=self.build_answer_quality("", [], [], verify_claims=False)
            evidence_panel = {"total_retrieved": 0, "total_selected": 0, "retrieval_coverage": {"documents": [], "unique_documents": 0, "page_range": {}}, "chunks": []}
            return "", None, None, [], risk_assessment, confidence, quality, disclaimer, evidence_panel, expanded_query
        
        
        
        system_prompt=self.template_parser.get("rag","system_prompt")
        
        
        reserved_chars = FOOTER_RESERVE_CHARS + len(query) + len(conversation_block) + 50
        selected_documents=self._select_documents_for_prompt(
            retrived_document=retrived_document,
            max_documents=limit or settings.ANSWER_TOP_K,
            max_context_chars=settings.MAX_CONTEXT_CHARS,
            reserved_chars=reserved_chars,
        )
        confidence=self._build_confidence(retrived_document, selected_documents, question=query)

        if not confidence["generation_allowed"]:
            refusals = (get_safety_config().get("refusals") or {})
            refusal = refusals.get(
                "insufficient_official_evidence",
                (
                    "I do not have enough official guideline evidence to answer this safely. "
                    "Please consult a qualified healthcare professional or ask a question "
                    "within the indexed asthma guideline scope."
                ),
            )
            sources=self._build_sources(selected_documents)
            quality=self.build_answer_quality(refusal, sources, selected_documents, verify_claims=False)
            evidence_panel = self.build_evidence_panel(retrived_document, selected_documents)
            return refusal, None, None, sources, risk_assessment, confidence, quality, disclaimer, evidence_panel, expanded_query

        documnets_prompts="\n".join([
                self._render_document_prompt(idx, doc)
            for idx,doc in enumerate(selected_documents)
        ])
        
        footer_prompt=self.template_parser.get("rag","footer_prompt")

        if risk_assessment["risk_level"] == "needs_caution":
            footer_prompt = (
                "SAFETY NOTE: The user appears to be asking about a specific person's "
                "symptoms. Your answer MUST begin by clearly stating that you cannot "
                "provide a diagnosis or prescribe medication without a clinical "
                "assessment, and that they should consult a qualified healthcare "
                "professional. Then you may share the relevant general guideline "
                "information from the documents above, with citations.\n\n"
                + footer_prompt
            )
        
        chat_history = [
        self.generation_client.construct_prompt(
            prompt=system_prompt,
            role=self.generation_client.enums.SYSTEM.value,
                        )
                    ]

        full_prompt = "\n\n".join(filter(None, [
            documnets_prompts,
            conversation_block,
            f"## User Question:\n{query}",
            footer_prompt,
        ]))

        answer=await self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        if risk_assessment["risk_level"] == "needs_caution" and answer:
            answer = (
                "I can't diagnose this person or prescribe medication without a "
                "clinical assessment. Please consult a qualified healthcare "
                "professional. Here is general information from the guidelines:\n\n"
            ) + answer

        sources=self._build_sources(selected_documents) if include_sources else []
        quality=self.build_answer_quality(answer or "", sources, selected_documents, verify_claims=verify_claims)

        max_regenerations = settings.ANSWER_MAX_VERIFICATION_REGENERATIONS
        if verify_claims and answer and max_regenerations > 0:
            for attempt in range(1, max_regenerations + 1):
                if quality["citation_faithfulness"] >= 1.0 and quality["unsupported_claim_rate"] == 0.0:
                    break

                correction_footer = (
                    "IMPORTANT: Your previous answer draft failed automated verification: "
                    "every factual claim must carry the citation [Document Name, p. PAGE] using the "
                    "EXACT document names from the '## Document Name' headers above. "
                    "Rewrite the answer from scratch, using ONLY the documents above, cite every claim, "
                    "and do not add pleasantries or restate the question."
                )
                chat_history = [
                    self.generation_client.construct_prompt(
                        prompt=system_prompt,
                        role=self.generation_client.enums.SYSTEM.value,
                    )
                ]
                full_prompt = "\n\n".join(filter(None, [
                    documnets_prompts,
                    conversation_block,
                    f"## User Question:\n{query}",
                    correction_footer,
                ]))
                retry_answer = await self.generation_client.generate_text(
                    prompt=full_prompt,
                    chat_history=chat_history
                )
                if not retry_answer:
                    break
                answer = retry_answer
                if risk_assessment["risk_level"] == "needs_caution":
                    answer = (
                        "I can't diagnose this person or prescribe medication without a "
                        "clinical assessment. Please consult a qualified healthcare "
                        "professional. Here is general information from the guidelines:\n\n"
                    ) + answer
                quality=self.build_answer_quality(answer, sources, selected_documents, verify_claims=verify_claims)

        evidence_panel = self.build_evidence_panel(retrived_document, selected_documents)
        confidence = self._apply_post_generation_confidence(confidence, quality)
        return answer , full_prompt , chat_history, sources, risk_assessment, confidence, quality, disclaimer, evidence_panel, expanded_query


    @staticmethod
    def _sse_event(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    async def answer_rag_question_stream(self, project: Project, query: str, limit: int = 5,
                                          score_threshold: float = None,
                                          metadata_filter: dict = None,
                                          include_sources: bool = True,
                                          retrieval_mode: str = "hybrid",
                                          rerank: bool = True,
                                          rerank_top_k: int = None,
                                          expand_query: bool = True,
                                          verify_claims: bool = True,
                                          conversation_history: list = None):
        settings = get_settings()
        disclaimer = self._clinical_disclaimer()

        yield self._sse_event("phase", {"phase": "classifying"})

        conversation_block = self._build_conversation_block(conversation_history)
        retrieval_query = self._build_retrieval_query(query, conversation_history)
        expanded_query = self.expand_query(retrieval_query) if expand_query else retrieval_query
        risk_assessment = self.classify_input_risk(query)
        refusal_answer = self._build_refusal_answer(risk_assessment)

        if risk_assessment["risk_level"] == "refuse_redirect":
            confidence = self._build_confidence([], [], question=query)
            confidence["generation_allowed"] = False
            confidence["reason"] = "blocked_by_safety_classifier"
            quality = self.build_answer_quality(refusal_answer, [], [], verify_claims=False)
            evidence_panel = {"total_retrieved": 0, "total_selected": 0, "retrieval_coverage": {"documents": [], "unique_documents": 0, "page_range": {}}, "chunks": []}
            yield self._sse_event("done", {
                "answer": refusal_answer, "sources": [], "risk_assessment": risk_assessment,
                "confidence": confidence, "quality": quality, "disclaimer": disclaimer,
                "evidence_panel": evidence_panel, "expanded_query": expanded_query,
            })
            return

        yield self._sse_event("phase", {"phase": "retrieving"})

        retrived_document = await self.search_vector_db_collection(
            project=project, text=retrieval_query,
            limit=max(limit, settings.RETRIEVAL_TOP_K),
            score_threshold=score_threshold if score_threshold is not None else settings.RETRIEVAL_SCORE_THRESHOLD,
            metadata_filter=metadata_filter, retrieval_mode=retrieval_mode,
            rerank=rerank, rerank_top_k=rerank_top_k or settings.RERANK_TOP_K,
            expand_query=expand_query,
        )

        if not retrived_document or len(retrived_document) == 0:
            confidence = self._build_confidence([], [], question=query)
            quality = self.build_answer_quality("", [], [], verify_claims=False)
            evidence_panel = {"total_retrieved": 0, "total_selected": 0, "retrieval_coverage": {"documents": [], "unique_documents": 0, "page_range": {}}, "chunks": []}
            yield self._sse_event("done", {
                "answer": "", "sources": [], "risk_assessment": risk_assessment,
                "confidence": confidence, "quality": quality, "disclaimer": disclaimer,
                "evidence_panel": evidence_panel, "expanded_query": expanded_query,
            })
            return

        if rerank:
            yield self._sse_event("phase", {"phase": "reranking"})

        yield self._sse_event("phase", {"phase": "checking_evidence"})

        system_prompt = self.template_parser.get("rag", "system_prompt")
        reserved_chars = FOOTER_RESERVE_CHARS + len(query) + len(conversation_block) + 50
        selected_documents = self._select_documents_for_prompt(
            retrived_document=retrived_document,
            max_documents=limit or settings.ANSWER_TOP_K,
            max_context_chars=settings.MAX_CONTEXT_CHARS,
            reserved_chars=reserved_chars,
        )
        confidence = self._build_confidence(retrived_document, selected_documents, question=query)

        if not confidence["generation_allowed"]:
            refusals = (get_safety_config().get("refusals") or {})
            refusal = refusals.get("insufficient_official_evidence",
                "I do not have enough official guideline evidence to answer this safely. "
                "Please consult a qualified healthcare professional or ask a question "
                "within the indexed asthma guideline scope.")
            sources = self._build_sources(selected_documents)
            quality = self.build_answer_quality(refusal, sources, selected_documents, verify_claims=False)
            evidence_panel = self.build_evidence_panel(retrived_document, selected_documents)
            yield self._sse_event("done", {
                "answer": refusal, "sources": sources, "risk_assessment": risk_assessment,
                "confidence": confidence, "quality": quality, "disclaimer": disclaimer,
                "evidence_panel": evidence_panel, "expanded_query": expanded_query,
            })
            return

        documnets_prompts = "\n".join([
            self._render_document_prompt(idx, doc)
            for idx, doc in enumerate(selected_documents)
        ])
        footer_prompt = self.template_parser.get("rag", "footer_prompt")

        if risk_assessment["risk_level"] == "needs_caution":
            footer_prompt = (
                "SAFETY NOTE: The user appears to be asking about a specific person's "
                "symptoms. Your answer MUST begin by clearly stating that you cannot "
                "provide a diagnosis or prescribe medication without a clinical "
                "assessment, and that they should consult a qualified healthcare "
                "professional. Then you may share the relevant general guideline "
                "information from the documents above, with citations.\n\n"
                + footer_prompt
            )

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]
        full_prompt = "\n\n".join(filter(None, [
            documnets_prompts, conversation_block,
            f"## User Question:\n{query}", footer_prompt,
        ]))

        yield self._sse_event("phase", {"phase": "generating"})

        answer = ""
        try:
            async for token in self.generation_client.generate_text_stream(
                prompt=full_prompt, chat_history=chat_history
            ):
                answer += token
                yield self._sse_event("token", {"token": token})
        except Exception:
            answer = None

        if answer is None:
            answer = await self.generation_client.generate_text(
                prompt=full_prompt, chat_history=chat_history
            )
            if answer:
                yield self._sse_event("phase", {"phase": "generating"})
                yield self._sse_event("token", {"token": answer, "full": True})

        if risk_assessment["risk_level"] == "needs_caution" and answer:
            answer = (
                "I can't diagnose this person or prescribe medication without a "
                "clinical assessment. Please consult a qualified healthcare "
                "professional. Here is general information from the guidelines:\n\n"
            ) + answer

        sources = self._build_sources(selected_documents) if include_sources else []
        yield self._sse_event("phase", {"phase": "verifying"})
        quality = self.build_answer_quality(answer or "", sources, selected_documents, verify_claims=verify_claims)

        max_regenerations = settings.ANSWER_MAX_VERIFICATION_REGENERATIONS
        if verify_claims and answer and max_regenerations > 0:
            for attempt in range(1, max_regenerations + 1):
                if quality["citation_faithfulness"] >= 1.0 and quality["unsupported_claim_rate"] == 0.0:
                    break

                yield self._sse_event("phase", {"phase": "regenerating"})

                correction_footer = (
                    "IMPORTANT: Your previous answer draft failed automated verification: "
                    "every factual claim must carry the citation [Document Name, p. PAGE] using the "
                    "EXACT document names from the '## Document Name' headers above. "
                    "Rewrite the answer from scratch, using ONLY the documents above, cite every claim, "
                    "and do not add pleasantries or restate the question."
                )
                chat_history = [
                    self.generation_client.construct_prompt(
                        prompt=system_prompt,
                        role=self.generation_client.enums.SYSTEM.value,
                    )
                ]
                full_prompt = "\n\n".join(filter(None, [
                    documnets_prompts, conversation_block,
                    f"## User Question:\n{query}", correction_footer,
                ]))
                retry_answer = await self.generation_client.generate_text(
                    prompt=full_prompt, chat_history=chat_history
                )
                if not retry_answer:
                    break
                answer = retry_answer
                if risk_assessment["risk_level"] == "needs_caution":
                    answer = (
                        "I can't diagnose this person or prescribe medication without a "
                        "clinical assessment. Please consult a qualified healthcare "
                        "professional. Here is general information from the guidelines:\n\n"
                    ) + answer
                quality = self.build_answer_quality(answer, sources, selected_documents, verify_claims=verify_claims)

        evidence_panel = self.build_evidence_panel(retrived_document, selected_documents)
        yield self._sse_event("done", {
            "answer": answer, "sources": sources, "risk_assessment": risk_assessment,
            "confidence": confidence, "quality": quality, "disclaimer": disclaimer,
            "evidence_panel": evidence_panel, "expanded_query": expanded_query,
        })

    CONVERSATION_CONTEXT_TURNS = 3
    CONVERSATION_ANSWER_CHARS = 400

    def _recent_conversation_turns(self, conversation_history: list):
        if not conversation_history:
            return []
        return [
            turn for turn in conversation_history[-self.CONVERSATION_CONTEXT_TURNS:]
            if (turn.get("question") or "").strip()
        ]

    def _build_retrieval_query(self, query: str, conversation_history: list):
        """Follow-up questions ("what about someone older?") carry no topic
        of their own, so prior turns are folded into the retrieval text —
        otherwise the vector/keyword search has nothing asthma-related to
        match against."""
        turns = self._recent_conversation_turns(conversation_history)
        if not turns:
            return query

        context = " ".join((turn.get("question") or "").strip() for turn in turns)
        return f"{context} {query}".strip()

    def _build_conversation_block(self, conversation_history: list):
        turns = self._recent_conversation_turns(conversation_history)
        if not turns:
            return ""

        lines = ["## Conversation so far (for context on follow-up questions):"]
        for turn in turns:
            question = (turn.get("question") or "").strip()
            answer = (turn.get("answer") or "").strip()
            lines.append(f"User: {question}")
            if answer:
                lines.append(f"Assistant: {answer[:self.CONVERSATION_ANSWER_CHARS]}")
        lines.append(
            "The user's new question below may be a follow-up referring back to this "
            "conversation (e.g. a pronoun, an age group, or an implied topic) — resolve "
            "that reference using the conversation above, then answer strictly from the "
            "documents."
        )
        return "\n".join(lines)

    def _parse_chunk_metadata(self, metadata):
        if not metadata:
            return {}

        if isinstance(metadata, dict):
            return metadata

        try:
            return json.loads(metadata)
        except Exception:
            return {}

    def expand_query(self, query: str):
        text=query or ""
        lowered=text.lower()
        expansions=[]

        for pat, expansion in self._EXPAND_PATTERNS:
            if pat.search(lowered):
                expansions.append(expansion)

        if not expansions:
            return text

        expanded_terms=" ".join(expansions)
        return f"{text}\n\nExpanded retrieval terms: {expanded_terms}"

    async def _build_rerank_candidates(self, collection_name: str, query: str, vector: list,
                                       limit: int, metadata_filter: dict = None,
                                       score_threshold: float = None,
                                       fallback_results: list = None):
        candidates = fallback_results or []
        return self._dedupe_documents([candidates])[:limit]

    @staticmethod
    def _dedupe_documents(candidate_groups: list):
        candidates=[]
        seen=set()

        for group in candidate_groups:
            for doc in group or []:
                if isinstance(doc, dict):
                    chunk_id=doc.get("chunk_id")
                    text=doc.get("text")
                else:
                    chunk_id=getattr(doc, "chunk_id", None)
                    text=getattr(doc, "text", None)
                key=chunk_id or text
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(doc)

        return candidates
    def _render_document_prompt(self, idx:int, doc:dict):
        metadata=doc.get("metadata") or {}
        return self.template_parser.get("rag","document_prompt",{
            "doc_num": idx+1,
            "doc_name": metadata.get("document_name") or metadata.get("file_name") or "Unknown",
            "page_number": metadata.get("page_number") or "",
            "section_title": metadata.get("section_title") or "",
            "source_url": metadata.get("source_url") or "",
            "chunk_text": self.generation_client.process_text(doc.get("text") or ""),
        })

    def _select_documents_for_prompt(self, retrived_document: list, max_documents: int,
                                    max_context_chars: int, reserved_chars: int = 0):
        budget = max_context_chars - reserved_chars
        selected_documents = []
        used_context_chars = 0
        seen_documents = set()

        for doc in retrived_document:
            metadata = doc.get("metadata") or {}
            dedupe_key = (metadata.get("asset_id"), metadata.get("chunk_order"), doc.get("text"))
            if dedupe_key in seen_documents:
                continue

            text = doc.get("text") or ""
            next_context_size = used_context_chars + len(text) + DOC_HEADER_OVERHEAD_CHARS

            if selected_documents and next_context_size > budget:
                continue

            seen_documents.add(dedupe_key)
            selected_documents.append(doc)
            used_context_chars += len(text) + DOC_HEADER_OVERHEAD_CHARS

            if len(selected_documents) >= max_documents:
                break

        return selected_documents

    # -- Delegates: behavior lives in focused collaborators (rag_safety,
    # answer_quality, confidence); these keep the public surface used by
    # routes, the answer pipeline, and tests unchanged.
    def classify_input_risk(self, query: str):
        return self.safety.classify_input_risk(query)

    def _build_refusal_answer(self, risk_assessment: dict):
        return self.safety._build_refusal_answer(risk_assessment)

    @staticmethod
    def _clinical_disclaimer():
        return SafetyClassifier._clinical_disclaimer()

    def _build_confidence(self, retrieved_documents: list, selected_documents: list, question: str = ""):
        # Resolved in this module's namespace so the test patch-point
        # (controllers.NLPController.get_settings) keeps working.
        return self.scorer._build_confidence(retrieved_documents, selected_documents, question=question, settings=get_settings())

    @classmethod
    def _apply_post_generation_confidence(cls, confidence: dict, quality: dict):
        return ConfidenceScorer._apply_post_generation_confidence(confidence, quality)

    @staticmethod
    def _build_confidence_summary(confidence: dict):
        return ConfidenceScorer._build_confidence_summary(confidence)

    def verify_citations(self, answer: str, sources: list):
        return self.evaluator.verify_citations(answer, sources)

    def build_answer_quality(self, answer: str, sources: list, selected_documents: list, verify_claims: bool = True):
        return self.evaluator.build_answer_quality(answer, sources, selected_documents, verify_claims=verify_claims)

    def detect_unsupported_claims(self, answer: str, selected_documents: list, citations: list):
        return self.evaluator.detect_unsupported_claims(answer, selected_documents, citations)

    def build_evidence_panel(self, retrieved_documents: list, selected_documents: list):
        return self.evaluator.build_evidence_panel(retrieved_documents, selected_documents)

    def _build_sources(self, selected_documents: list):
        return self.evaluator._build_sources(selected_documents)

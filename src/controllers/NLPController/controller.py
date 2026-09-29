import json

from controllers.BaseController import BaseController
from controllers.NLPController.retrieval import RetrievalService
from domain.confidence import ConfidenceScorer
from domain.contracts import AnswerRequest, AnswerResult
from domain.quality import AnswerQualityEvaluator
from domain.safety import SafetyClassifier
from helpers.config import get_settings
from helpers.safety_config import get_safety_config
from models.db_schemas import Project

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
        self.retrieval = RetrievalService(
            vectordb_client=vectordb_client,
            embedding_client=embedding_client,
            template_parser=template_parser,
            generation_client=generation_client,
            rerank_client=rerank_client,
        )

    @classmethod
    def from_app(cls, app) -> "NLPController":
        """Single construction site for the answering service (keeps routes thin)."""
        return cls(
            vectordb_client=app.vectordb_client,
            generation_client=app.generation_client,
            embedding_client=app.embedding_client,
            template_parser=app.template_parser,
            rerank_client=app.rerank_client,
        )

    async def answer(self, project: Project, request: AnswerRequest) -> AnswerResult:
        """Use-case entry point: validated request in, result object out.

        Default resolution (rerank / query expansion) lives here so routes
        stay free of pipeline policy.
        """
        use_rerank = request.rerank if request.rerank is not None else True
        use_expansion = request.expand_query if request.expand_query is not None else True
        (answer, full_prompt, chat_history, sources, risk_assessment,
         confidence, quality, disclaimer, evidence_panel,
         expanded_query) = await self.answer_rag_question(
            project=project,
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold,
            metadata_filter=request.metadata_filter,
            include_sources=request.include_sources,
            retrieval_mode=request.retrieval_mode,
            rerank=use_rerank,
            rerank_top_k=request.rerank_top_k,
            expand_query=use_expansion,
            verify_claims=request.verify_claims,
            conversation_history=request.conversation_history,
        )
        return AnswerResult(
            answer=answer,
            full_prompt=full_prompt,
            chat_history=chat_history,
            sources=sources,
            risk_assessment=risk_assessment,
            confidence=confidence,
            quality=quality,
            disclaimer=disclaimer,
            evidence_panel=evidence_panel,
            expanded_query=expanded_query,
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
        
        retrieved_documents= await self.search_vector_db_collection(
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
        
        if not retrieved_documents or len(retrieved_documents)==0:
            confidence=self._build_confidence([], [], question=query)
            quality=self.build_answer_quality("", [], [], verify_claims=False)
            evidence_panel = {"total_retrieved": 0, "total_selected": 0, "retrieval_coverage": {"documents": [], "unique_documents": 0, "page_range": {}}, "chunks": []}
            return "", None, None, [], risk_assessment, confidence, quality, disclaimer, evidence_panel, expanded_query
        
        
        
        system_prompt=self.template_parser.get("rag","system_prompt")
        
        
        reserved_chars = FOOTER_RESERVE_CHARS + len(query) + len(conversation_block) + 50
        selected_documents=self._select_documents_for_prompt(
            retrieved_documents=retrieved_documents,
            max_documents=limit or settings.ANSWER_TOP_K,
            max_context_chars=settings.MAX_CONTEXT_CHARS,
            reserved_chars=reserved_chars,
        )
        confidence=self._build_confidence(retrieved_documents, selected_documents, question=query)

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
            evidence_panel = self.build_evidence_panel(retrieved_documents, selected_documents)
            return refusal, None, None, sources, risk_assessment, confidence, quality, disclaimer, evidence_panel, expanded_query

        document_prompts="\n".join([
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
            document_prompts,
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
                    document_prompts,
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

        evidence_panel = self.build_evidence_panel(retrieved_documents, selected_documents)
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

        retrieved_documents = await self.search_vector_db_collection(
            project=project, text=retrieval_query,
            limit=max(limit, settings.RETRIEVAL_TOP_K),
            score_threshold=score_threshold if score_threshold is not None else settings.RETRIEVAL_SCORE_THRESHOLD,
            metadata_filter=metadata_filter, retrieval_mode=retrieval_mode,
            rerank=rerank, rerank_top_k=rerank_top_k or settings.RERANK_TOP_K,
            expand_query=expand_query,
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
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
            retrieved_documents=retrieved_documents,
            max_documents=limit or settings.ANSWER_TOP_K,
            max_context_chars=settings.MAX_CONTEXT_CHARS,
            reserved_chars=reserved_chars,
        )
        confidence = self._build_confidence(retrieved_documents, selected_documents, question=query)

        if not confidence["generation_allowed"]:
            refusals = (get_safety_config().get("refusals") or {})
            refusal = refusals.get("insufficient_official_evidence",
                "I do not have enough official guideline evidence to answer this safely. "
                "Please consult a qualified healthcare professional or ask a question "
                "within the indexed asthma guideline scope.")
            sources = self._build_sources(selected_documents)
            quality = self.build_answer_quality(refusal, sources, selected_documents, verify_claims=False)
            evidence_panel = self.build_evidence_panel(retrieved_documents, selected_documents)
            yield self._sse_event("done", {
                "answer": refusal, "sources": sources, "risk_assessment": risk_assessment,
                "confidence": confidence, "quality": quality, "disclaimer": disclaimer,
                "evidence_panel": evidence_panel, "expanded_query": expanded_query,
            })
            return

        document_prompts = "\n".join([
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
            document_prompts, conversation_block,
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
                    document_prompts, conversation_block,
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

        evidence_panel = self.build_evidence_panel(retrieved_documents, selected_documents)
        yield self._sse_event("done", {
            "answer": answer, "sources": sources, "risk_assessment": risk_assessment,
            "confidence": confidence, "quality": quality, "disclaimer": disclaimer,
            "evidence_panel": evidence_panel, "expanded_query": expanded_query,
        })

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
        from controllers.NLPController import get_settings as _pkg_settings  # package attr: patch-point
        return self.scorer._build_confidence(retrieved_documents, selected_documents, question=question, settings=_pkg_settings())

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

    # -- Retrieval delegates: capability lives on RetrievalService;
    # these keep routes and the answer pipeline unchanged.
    def create_collection_name(self, *args, **kwargs):
        return self.retrieval.create_collection_name(*args, **kwargs)
    async def reset_vector_db_collection(self, *args, **kwargs):
        return await self.retrieval.reset_vector_db_collection(*args, **kwargs)
    async def get_vector_db_collection_info(self, *args, **kwargs):
        return await self.retrieval.get_vector_db_collection_info(*args, **kwargs)
    async def index_into_vector_db(self, *args, **kwargs):
        return await self.retrieval.index_into_vector_db(*args, **kwargs)
    async def search_vector_db_collection(self, *args, **kwargs):
        return await self.retrieval.search_vector_db_collection(*args, **kwargs)
    def _recent_conversation_turns(self, *args, **kwargs):
        return self.retrieval._recent_conversation_turns(*args, **kwargs)
    def _build_retrieval_query(self, *args, **kwargs):
        return self.retrieval._build_retrieval_query(*args, **kwargs)
    def _build_conversation_block(self, *args, **kwargs):
        return self.retrieval._build_conversation_block(*args, **kwargs)
    def _parse_chunk_metadata(self, *args, **kwargs):
        return self.retrieval._parse_chunk_metadata(*args, **kwargs)
    def expand_query(self, *args, **kwargs):
        return self.retrieval.expand_query(*args, **kwargs)
    async def _build_rerank_candidates(self, *args, **kwargs):
        return await self.retrieval._build_rerank_candidates(*args, **kwargs)
    @staticmethod
    def _dedupe_documents(*args, **kwargs):
        return RetrievalService._dedupe_documents(*args, **kwargs)
    def _render_document_prompt(self, *args, **kwargs):
        return self.retrieval._render_document_prompt(*args, **kwargs)
    def _select_documents_for_prompt(self, *args, **kwargs):
        return self.retrieval._select_documents_for_prompt(*args, **kwargs)

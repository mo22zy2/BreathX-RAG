from .RerankEnums import RerankType
from .providers import CoHereRerankProvider
from domain.exceptions import UpstreamError
from helpers.config import Settings


class RerankProviderFactory:

    def __init__(self, config: Settings):
        self.config = config

    def create_provider(self, provider: str):
        if provider == RerankType.COHERE.value:
            return CoHereRerankProvider(
                api_key=self.config.COHERE_API_KEY,
                model_id=self.config.RERANK_MODEL_ID,
            )

        raise UpstreamError(f"unknown rerank backend: {provider!r}")

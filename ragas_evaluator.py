from typing import Dict, List, Optional

try:
    from ragas import SingleTurnSample, evaluate
    from ragas.metrics import (
        ResponseRelevancy,
        Faithfulness,
        BleuScore,
        RougeScore,
        NonLLMContextPrecisionWithReference,
    )
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

PRIMARY_METRICS = [
    ResponseRelevancy(),
    Faithfulness(),
]

ADDITIONAL_METRICS = [
    NonLLMContextPrecisionWithReference(),
    BleuScore(),
    RougeScore(),
]

def _build_llm_wrappers(openai_api_key: str):
    """Create LangChain wrappers for RAGAS evaluators."""
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        max_retries=2,
    )
    embeddings = OpenAIEmbeddings(
        api_key=openai_api_key,
        model="text-embedding-3-small",
    )
    return LangchainLLMWrapper(langchain_llm=llm), LangchainEmbeddingsWrapper(embeddings=embeddings)

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    from os import getenv

    api_key = (
        getenv("OPENAI_API_KEY")
        or getenv("CHROMA_OPENAI_API_KEY")
        or getenv("OPENAI_KEY")
    )
    if not api_key:
        return {"error": "Missing OPENAI_API_KEY for RAGAS evaluation"}

    llm_wrapper, embedding_wrapper = _build_llm_wrappers(api_key)

    metrics = PRIMARY_METRICS + ADDITIONAL_METRICS

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts or [],
        reference_contexts=contexts or [],
        reference=" ".join(contexts) if contexts else "",
    )
    from ragas import EvaluationDataset
    dataset = EvaluationDataset(samples=[sample])

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm_wrapper,
            embeddings=embedding_wrapper,
        )
    except Exception as exc:
        return {"error": f"RAGAS evaluation failed: {exc}"}

    scores: Dict[str, float] = {}
    if hasattr(result, "scores"):
        data = result.scores
        if isinstance(data, list) and data:
            first_entry = data[0]
            if isinstance(first_entry, dict):
                for metric_name, value in first_entry.items():
                    try:
                        scores[metric_name] = float(value)
                    except (TypeError, ValueError):
                        continue
        elif hasattr(data, "to_dict"):
            records = data.to_dict(orient="records")
            if records:
                for metric_name, value in records[0].items():
                    try:
                        scores[metric_name] = float(value)
                    except (TypeError, ValueError):
                        continue

    return scores or {"error": "RAGAS evaluation produced no scores"}

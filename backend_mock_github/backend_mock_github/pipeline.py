"""End-to-end orchestration for finance intern assessment and action planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from database import AssessmentRepository, AssessmentResult
from llm_generator import LLMClient, MockLLMClient, build_prompt
from rag_retriever import IndexedPolicyRetriever, KnowledgeDocument, Retriever
from rule_engine import AssessmentDecision, InternAssessment, evaluate_assessment


@dataclass(frozen=True)
class PipelineOutput:
    decision: AssessmentDecision
    persisted_result: AssessmentResult
    retrieved_documents: tuple[KnowledgeDocument, ...]
    prompt: str
    action_plan: str


class InternEvaluationPipeline:
    """Executes rule evaluation, persistence, retrieval, and plan generation in order."""

    def __init__(self, repository: AssessmentRepository, retriever: Retriever) -> None:
        self.repository = repository
        self.retriever = retriever
        self.repository.initialize()

    def run(self, assessment: InternAssessment, llm_client: LLMClient | None = None) -> PipelineOutput:
        """Process one assessment through all three stages."""
        decision = evaluate_assessment(assessment)
        persisted_result = self.repository.save(decision)
        documents = tuple(self.retriever.retrieve(decision.weak_topics, decision.priority))
        prompt = build_prompt(decision, documents)
        client = llm_client or MockLLMClient(decision, documents)
        action_plan = client.generate(prompt)
        return PipelineOutput(decision, persisted_result, documents, prompt, action_plan)


def main() -> None:
    """Run the complete pipeline using the indexed approved knowledge base."""
    database_path = Path(__file__).with_name("intern_assessments.db")
    index_path = Path(__file__).with_name("data") / "policy_vectors.json"
    pipeline = InternEvaluationPipeline(
        AssessmentRepository(database_path),
        IndexedPolicyRetriever(index_path),
    )
    samples: Sequence[InternAssessment] = (
        InternAssessment("INT-001", "Maya Patel", 0.42, ("Excel", "Financial Modeling")),
        InternAssessment("INT-002", "Omar Hassan", 0.64, ("Valuation", "DCF")),
        InternAssessment("INT-003", "Leah Kim", 0.84, ("Cash Flow", "Financial Statements")),
    )
    for assessment in samples:
        output = pipeline.run(assessment)
        print("=" * 72)
        print(output.action_plan)
        print(f"Retrieved sources: {', '.join(document.title for document in output.retrieved_documents) or 'None'}")
        print(f"Saved assessment_results.id: {output.persisted_result.id}")


if __name__ == "__main__":
    main()

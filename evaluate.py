"""
Quality Evaluation for Indecimal RAG System

This module provides:
1. 15 test questions derived from the documents
2. Automated evaluation of retrieval relevance and answer quality
3. Analysis of hallucinations and groundedness
"""

import json
import time
from typing import List, Dict
from rag_pipeline import RAGPipeline


# Test questions derived from the three documents
TEST_QUESTIONS = [
    # From doc1.md - Company Overview & Customer Journey
    {
        "question": "What does Indecimal promise to provide for home construction?",
        "expected_source": "doc1.md",
        "expected_topics": ["confidence", "commitment", "transparent pricing", "quality assurance"]
    },
    {
        "question": "What are the 5 operating principles of Indecimal?",
        "expected_source": "doc1.md",
        "expected_topics": ["smooth construction", "best pricing", "quality assurance", "stage-based payments", "transparent tracking"]
    },
    {
        "question": "How many quality checks does Indecimal perform?",
        "expected_source": "doc1.md",
        "expected_topics": ["445", "quality assurance"]
    },
    {
        "question": "What are the 10 steps in Indecimal's customer journey?",
        "expected_source": "doc1.md",
        "expected_topics": ["raise request", "meet experts", "financing", "design", "plans", "book", "construction", "interior", "move in", "maintenance"]
    },
    {
        "question": "How does Indecimal provide real-time progress visibility?",
        "expected_source": "doc1.md",
        "expected_topics": ["app", "dashboard", "photo updates", "tracking"]
    },
    
    # From doc2.md - Package Comparison & Specifications
    {
        "question": "What is the price per sqft for the Premier package?",
        "expected_source": "doc2.md",
        "expected_topics": ["1,995", "sqft", "GST"]
    },
    {
        "question": "What brands of steel are used in the Pinnacle package?",
        "expected_source": "doc2.md",
        "expected_topics": ["TATA", "80,000"]
    },
    {
        "question": "What is the ceiling height for all packages?",
        "expected_source": "doc2.md",
        "expected_topics": ["10 ft", "floor-to-floor"]
    },
    {
        "question": "What are the bathroom fittings allowances for the Infinia package?",
        "expected_source": "doc2.md",
        "expected_topics": ["70,000", "Jaquar", "Essco"]
    },
    {
        "question": "What type of paint is used for exterior in the Pinnacle package?",
        "expected_source": "doc2.md",
        "expected_topics": ["Asian Paints", "Apex Ultima"]
    },
    
    # From doc3.md - Policies & Guarantees
    {
        "question": "How does the escrow-based payment model work?",
        "expected_source": "doc3.md",
        "expected_topics": ["escrow", "project manager", "verification", "disbursed"]
    },
    {
        "question": "What mechanisms does Indecimal use to prevent construction delays?",
        "expected_source": "doc3.md",
        "expected_topics": ["project management", "daily tracking", "flagging", "penalisation", "automated task"]
    },
    {
        "question": "What does the zero cost maintenance program cover?",
        "expected_source": "doc3.md",
        "expected_topics": ["plumbing", "electrical", "painting", "roofing", "masonry"]
    },
    {
        "question": "How long does home financing confirmation take?",
        "expected_source": "doc3.md",
        "expected_topics": ["7 days", "30 days", "disbursal"]
    },
    {
        "question": "What is the partner onboarding process?",
        "expected_source": "doc3.md",
        "expected_topics": ["verification", "background", "financial", "agreement", "SOP"]
    }
]


def evaluate_retrieval(result: Dict, test_case: Dict) -> Dict:
    """
    Evaluate retrieval quality for a single test case.
    
    Checks:
    1. Was the expected source document retrieved?
    2. Were relevant topics covered in retrieved chunks?
    """
    retrieved_chunks = result["retrieved_chunks"]
    
    # Check if expected source is in retrieved chunks
    sources_retrieved = [chunk["source"] for chunk in retrieved_chunks]
    expected_source_found = any(test_case["expected_source"] in s for s in sources_retrieved)
    
    # Check topic coverage
    all_chunk_text = " ".join([chunk["text"].lower() for chunk in retrieved_chunks])
    topics_found = []
    topics_missing = []
    
    for topic in test_case["expected_topics"]:
        if topic.lower() in all_chunk_text:
            topics_found.append(topic)
        else:
            topics_missing.append(topic)
    
    topic_coverage = len(topics_found) / len(test_case["expected_topics"]) if test_case["expected_topics"] else 0
    
    return {
        "expected_source_found": expected_source_found,
        "sources_retrieved": sources_retrieved,
        "topic_coverage": round(topic_coverage, 2),
        "topics_found": topics_found,
        "topics_missing": topics_missing
    }


def evaluate_answer(result: Dict, test_case: Dict) -> Dict:
    """
    Evaluate answer quality for a single test case.
    
    Checks:
    1. Does the answer reference expected topics?
    2. Is the answer grounded (contains source citations)?
    3. Does it avoid clear hallucinations?
    """
    answer = result["generated_answer"].lower()
    
    # Check topic coverage in answer
    topics_in_answer = []
    for topic in test_case["expected_topics"]:
        if topic.lower() in answer:
            topics_in_answer.append(topic)
    
    topic_coverage = len(topics_in_answer) / len(test_case["expected_topics"]) if test_case["expected_topics"] else 0
    
    # Check for source citations (grounding indicator)
    has_citations = any(marker in answer for marker in ["source", "[", "doc1", "doc2", "doc3"])
    
    # Check for hallucination phrases
    hallucination_phrases = [
        "i don't have information",
        "cannot be found in the context",
        "not mentioned in",
        "i'm not sure"
    ]
    indicates_no_info = any(phrase in answer for phrase in hallucination_phrases)
    
    return {
        "answer_topic_coverage": round(topic_coverage, 2),
        "topics_in_answer": topics_in_answer,
        "has_citations": has_citations,
        "indicates_no_info": indicates_no_info,
        "answer_length": len(answer.split())
    }


def run_evaluation(rag: RAGPipeline = None) -> Dict:
    """
    Run full evaluation on all test questions.
    
    Returns comprehensive evaluation results.
    """
    if rag is None:
        print("Initializing RAG pipeline...")
        rag = RAGPipeline()
    
    results = []
    total_time = 0
    
    print(f"\nRunning evaluation on {len(TEST_QUESTIONS)} test questions...\n")
    print("=" * 70)
    
    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        question = test_case["question"]
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question}")
        
        # Time the query
        start_time = time.time()
        result = rag.query(question)
        query_time = time.time() - start_time
        total_time += query_time
        
        # Evaluate
        retrieval_eval = evaluate_retrieval(result, test_case)
        answer_eval = evaluate_answer(result, test_case)
        
        test_result = {
            "question": question,
            "expected_source": test_case["expected_source"],
            "query_time_seconds": round(query_time, 2),
            "retrieval": retrieval_eval,
            "answer": answer_eval,
            "generated_answer_preview": result["generated_answer"][:200] + "..."
        }
        results.append(test_result)
        
        # Print brief result
        status = "✅" if retrieval_eval["expected_source_found"] and answer_eval["answer_topic_coverage"] > 0.3 else "⚠️"
        print(f"   {status} Source: {'Found' if retrieval_eval['expected_source_found'] else 'Missing'} | "
              f"Topic Coverage: {answer_eval['answer_topic_coverage']:.0%} | "
              f"Time: {query_time:.2f}s")
    
    # Calculate summary statistics
    summary = {
        "total_questions": len(TEST_QUESTIONS),
        "avg_query_time_seconds": round(total_time / len(TEST_QUESTIONS), 2),
        "total_time_seconds": round(total_time, 2),
        "source_retrieval_accuracy": sum(1 for r in results if r["retrieval"]["expected_source_found"]) / len(results),
        "avg_topic_coverage_retrieval": sum(r["retrieval"]["topic_coverage"] for r in results) / len(results),
        "avg_topic_coverage_answer": sum(r["answer"]["answer_topic_coverage"] for r in results) / len(results),
        "answers_with_citations": sum(1 for r in results if r["answer"]["has_citations"]) / len(results),
        "answers_indicating_no_info": sum(1 for r in results if r["answer"]["indicates_no_info"]) / len(results)
    }
    
    print("\n" + "=" * 70)
    print("\n📊 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Average Query Time: {summary['avg_query_time_seconds']}s")
    print(f"Source Retrieval Accuracy: {summary['source_retrieval_accuracy']:.0%}")
    print(f"Avg Topic Coverage (Retrieval): {summary['avg_topic_coverage_retrieval']:.0%}")
    print(f"Avg Topic Coverage (Answer): {summary['avg_topic_coverage_answer']:.0%}")
    print(f"Answers with Citations: {summary['answers_with_citations']:.0%}")
    print(f"Answers Indicating No Info: {summary['answers_indicating_no_info']:.0%}")
    
    return {
        "summary": summary,
        "detailed_results": results
    }


def save_evaluation_report(evaluation_results: Dict, output_file: str = "evaluation_report.json"):
    """Save evaluation results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Evaluation report saved to: {output_file}")


if __name__ == "__main__":
    # Run evaluation
    results = run_evaluation()
    
    # Save report
    save_evaluation_report(results)
    
    # Print observations
    print("\n" + "=" * 70)
    print("📝 KEY OBSERVATIONS")
    print("=" * 70)
    
    summary = results["summary"]
    
    observations = []
    
    if summary["source_retrieval_accuracy"] >= 0.8:
        observations.append("✅ Strong retrieval accuracy - the correct source documents are being found")
    else:
        observations.append("⚠️ Retrieval accuracy could be improved - consider adjusting chunking strategy")
    
    if summary["avg_topic_coverage_answer"] >= 0.5:
        observations.append("✅ Good topic coverage in answers - key information is being captured")
    else:
        observations.append("⚠️ Topic coverage in answers is low - may need to retrieve more chunks")
    
    if summary["answers_with_citations"] >= 0.7:
        observations.append("✅ Model is providing citations - good for transparency")
    else:
        observations.append("⚠️ Citations are sparse - consider strengthening the grounding prompt")
    
    if summary["answers_indicating_no_info"] <= 0.1:
        observations.append("✅ Model rarely claims lack of information - documents are comprehensive")
    else:
        observations.append("⚠️ Model frequently indicates missing information - may need better retrieval")
    
    for obs in observations:
        print(f"\n{obs}")
    
    print("\n" + "=" * 70)

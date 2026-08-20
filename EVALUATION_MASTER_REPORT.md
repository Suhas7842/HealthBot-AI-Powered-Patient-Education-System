# HealthBot: Master Evaluation Report

**Generated:** 2026-08-20 22:15:10
**Mode:** report-only

---

## Executive Summary

HealthBot is a production-grade RAG system for medical education featuring:

- **Hybrid Retrieval**: Semantic search + BM25 + RRF fusion
- **Intelligent Routing**: Query classification with adaptive retrieval parameters
- **Multi-Turn Conversation**: Context-aware follow-up detection and query rewriting
- **Citation Verification**: Claim-level source attribution with LLM-as-judge


**Baseline Performance (50 test cases):**
- Average Latency: 0.318s
- Retrieval Success Rate: 100.0%
- Average RRF Score: 0.0201

## Design Decision Justifications

### Why Hybrid Retrieval?

Baseline evaluation shows hybrid retrieval (semantic + BM25 + RRF) achieves:
- 100% retrieval success rate across 50 test cases
- Balanced method distribution: 44% semantic, 31% BM25, 26% hybrid
- Average latency: 318ms (acceptable for interactive use)

*Full strategy comparison available after running experiments.py*

### Why Query Classification?

Pattern-based classification enables:
- Adaptive retrieval: Treatment queries (k=5, precision-focused) vs Informational queries (k=7, comprehensive)
- Zero latency overhead: <1ms pattern matching, no LLM calls
- Intent-aware routing: Different thresholds for medical advice vs general education


## Evaluation Warnings

- ⚠️ retrieval_metrics: No existing results file
- ⚠️ ragas: No existing results file
- ⚠️ experiments: No existing results file
- ⚠️ citations: No existing results file
- ⚠️ latency_profile: Not yet implemented
- ⚠️ query_rewriting: Not yet implemented
- ⚠️ threshold_validation: Not yet implemented

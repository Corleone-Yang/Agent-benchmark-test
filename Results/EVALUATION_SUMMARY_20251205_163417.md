# Qwen2.5-3B-Instruct Full Evaluation Results

**Model**: Qwen/Qwen2.5-3B-Instruct
**Test Date**: 2025-12-05 16:34:17
**Evaluation Method**: Manual judgment for each response

---

## Task Success Rates

| Task | Total | Correct | Success Rate |
|------|-------|---------|--------------|
| Math Reasoning | 10 | 10 | **100.0%** |
| Common Sense QA | 10 | 2 | **20.0%** |
| SQL Generation | 10 | 0 | **0.0%** |
| Knowledge Graph | 10 | 1 | **10.0%** |

---

## Summary

**Overall Performance:**
- Total questions tested: 40
- Total correct: 13
- Average success rate: 32.5%

**Task-by-Task Analysis:**

1. **Math Reasoning (100.0%)**
   - 10/10 problems solved correctly
   - Task: Basic arithmetic and word problems

2. **Common Sense QA (20.0%)**
   - 2/10 questions answered correctly
   - Task: Multiple choice common sense reasoning

3. **SQL Generation (0.0%)**
   - 0/10 queries generated correctly
   - Task: Natural language to SQL translation

4. **Knowledge Graph (10.0%)**
   - 1/10 entities identified correctly
   - Task: Multi-hop reasoning over knowledge graphs

---

## Result Files

- `math_reasoning_20251205_163417.json`
- `common_sense_qa_20251205_163417.json`
- `sql_generation_20251205_163417.json`
- `knowledge_graph_20251205_163417.json`

*Generated: 2025-12-05T16:34:17.237311*

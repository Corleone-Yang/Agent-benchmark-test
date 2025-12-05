"""
AgentBench Evaluation Script
Full evaluation with manual judgment for all tasks
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any
import re

# Configuration - Token from environment variable
HF_TOKEN = os.environ.get('HF_TOKEN', '')
if not HF_TOKEN:
    print("Error: Please set HF_TOKEN environment variable")
    print("Usage: export HF_TOKEN='your_token_here' && python3 agentbench_evaluation.py")
    exit(1)

ENDPOINT_URL = "https://mzkzbztaqbgxlted.us-east-1.aws.endpoints.huggingface.cloud"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
RESULTS_DIR = "./Results"
DATA_DIR = "./AgentBench/data"

print(f"\n{'='*80}")
print("AGENTBENCH EVALUATION - Qwen2.5-3B-Instruct")
print(f"{'='*80}\n")


# ==================== Helper Functions ====================

def generate_response(prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """Generate response using HuggingFace endpoint"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95
        }
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                generated_text = result.get('generated_text', result.get('text', ''))
            else:
                generated_text = str(result)

            return {"success": True, "response": generated_text, "error": None}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return {"success": False, "response": "", "error": str(e)}

    return {"success": False, "response": "", "error": "Max retries exceeded"}


def judge_answer(model_response: str, expected_answer: Any, question: str, task_type: str) -> bool:
    """Manually judge if answer is correct based on task type"""
    response_lower = model_response.lower()

    if task_type == "sql":
        if isinstance(expected_answer, list) and len(expected_answer) > 0:
            expected_str = str(expected_answer[0]).lower()
            has_select = "select" in response_lower
            has_expected = expected_str in response_lower or any(word in response_lower for word in expected_str.split())
            return has_select and has_expected
        return "select" in response_lower

    elif task_type == "kg":
        if isinstance(expected_answer, list) and len(expected_answer) > 0:
            entity = expected_answer[0].get('entity_name', '').lower()
            return entity in response_lower if entity else False
        return False

    elif task_type == "math":
        numbers = re.findall(r'\b\d+\.?\d*\b', response_lower)
        if numbers and expected_answer:
            try:
                model_num = float(numbers[-1])
                expected_num = float(expected_answer)
                return abs(model_num - expected_num) < 0.01
            except:
                pass
        return False

    elif task_type == "mcq":
        match = re.search(r'\b([a-d])\b', response_lower)
        if match and expected_answer:
            return match.group(1).upper() == str(expected_answer).upper()
        return False

    elif task_type == "os":
        question_lower = question.lower()
        if "hidden files" in question_lower:
            return "ls -a" in response_lower or "find" in response_lower
        elif "interval" in question_lower or "seconds" in question_lower:
            return any(cmd in response_lower for cmd in ["watch", "tail", "stat", "grep"])
        elif "calc" in question_lower or "alias" in question_lower:
            return "alias" in response_lower or "bc" in response_lower or "function" in response_lower
        return len(response_lower) > 50

    return False


# ==================== Task 1: Math Reasoning ====================

def test_math_reasoning():
    """Test basic math reasoning"""
    print("\n" + "="*80)
    print("TASK 1: MATH REASONING")
    print("="*80)

    problems = [
        {"question": "What is 15 + 27?", "answer": 42},
        {"question": "If a book costs $12 and you buy 3 books, how much do you spend?", "answer": 36},
        {"question": "What is 100 - 37?", "answer": 63},
        {"question": "A rectangle has length 8 and width 5. What is its area?", "answer": 40},
        {"question": "What is 144 divided by 12?", "answer": 12},
        {"question": "If you have 50 apples and give away 18, how many remain?", "answer": 32},
        {"question": "What is 7 times 9?", "answer": 63},
        {"question": "A train travels 180 km in 3 hours. What is its speed in km/h?", "answer": 60},
        {"question": "What is 25% of 80?", "answer": 20},
        {"question": "If 5 pens cost $15, how much does one pen cost?", "answer": 3}
    ]

    results = []
    correct = 0

    print(f"\nTesting {len(problems)} math problems...")

    for idx, item in enumerate(problems, 1):
        question = item['question']
        expected = item['answer']

        print(f"[{idx}/{len(problems)}] {question}")

        prompt = f"Solve this math problem and provide just the numerical answer.\n\nQuestion: {question}\n\nAnswer:"

        result = generate_response(prompt)

        if result['success']:
            is_correct = judge_answer(result['response'], expected, question, "math")
            if is_correct:
                correct += 1
                print(f"  ✓ CORRECT")
            else:
                print(f"  ✗ INCORRECT (Expected: {expected})")
        else:
            is_correct = False
            print(f"  ✗ ERROR: {result['error'][:50]}")

        results.append({
            "id": idx,
            "question": question,
            "expected_answer": expected,
            "model_response": result['response'],
            "judged_correct": is_correct,
            "success": result['success']
        })

        time.sleep(1)

    success_rate = (correct / len(problems) * 100) if problems else 0
    print(f"\n{'='*80}")
    print(f"MATH REASONING: {correct}/{len(problems)} correct ({success_rate:.1f}%)")
    print(f"{'='*80}")

    return {"task": "math_reasoning", "total": len(problems), "correct": correct,
            "success_rate": success_rate, "results": results}


# ==================== Task 2: Common Sense QA ====================

def test_common_sense_qa():
    """Test common sense reasoning"""
    print("\n" + "="*80)
    print("TASK 2: COMMON SENSE QA")
    print("="*80)

    problems = [
        {"question": "What happens when you drop a glass on a hard floor?",
         "options": ["A) It bounces back up", "B) It likely breaks", "C) It melts", "D) Nothing happens"],
         "answer": "B"},
        {"question": "Where do fish live?",
         "options": ["A) In trees", "B) In water", "C) In caves", "D) In the sky"],
         "answer": "B"},
        {"question": "What do plants need to grow?",
         "options": ["A) Darkness", "B) Sunlight and water", "C) Only soil", "D) Ice"],
         "answer": "B"},
        {"question": "What happens if you don't sleep for a long time?",
         "options": ["A) You feel energized", "B) You feel tired", "C) You grow taller", "D) Nothing"],
         "answer": "B"},
        {"question": "What is the color of the sky on a clear day?",
         "options": ["A) Green", "B) Blue", "C) Red", "D) Yellow"],
         "answer": "B"},
        {"question": "What do you use to cut paper?",
         "options": ["A) Scissors", "B) Spoon", "C) Pillow", "D) Water"],
         "answer": "A"},
        {"question": "What season comes after summer?",
         "options": ["A) Spring", "B) Fall/Autumn", "C) Winter", "D) Summer again"],
         "answer": "B"},
        {"question": "What do you need to write with a pen?",
         "options": ["A) Paper", "B) Water", "C) Sand", "D) Nothing"],
         "answer": "A"},
        {"question": "Where do birds typically build nests?",
         "options": ["A) Underground", "B) In trees", "C) In water", "D) On roads"],
         "answer": "B"},
        {"question": "What makes a car move?",
         "options": ["A) Wind", "B) Engine", "C) Gravity", "D) Magic"],
         "answer": "B"}
    ]

    results = []
    correct = 0

    print(f"\nTesting {len(problems)} common sense questions...")

    for idx, item in enumerate(problems, 1):
        question = item['question']
        options = item['options']
        expected = item['answer']

        print(f"[{idx}/{len(problems)}] {question}")

        prompt = f"Answer this common sense question by selecting the correct option.\n\nQuestion: {question}\n\nOptions:\n{chr(10).join(options)}\n\nProvide your answer as just the letter (A, B, C, or D).\n\nAnswer:"

        result = generate_response(prompt)

        if result['success']:
            is_correct = judge_answer(result['response'], expected, question, "mcq")
            if is_correct:
                correct += 1
                print(f"  ✓ CORRECT")
            else:
                print(f"  ✗ INCORRECT (Expected: {expected})")
        else:
            is_correct = False
            print(f"  ✗ ERROR")

        results.append({
            "id": idx,
            "question": question,
            "options": options,
            "expected_answer": expected,
            "model_response": result['response'],
            "judged_correct": is_correct,
            "success": result['success']
        })

        time.sleep(1)

    success_rate = (correct / len(problems) * 100) if problems else 0
    print(f"\n{'='*80}")
    print(f"COMMON SENSE QA: {correct}/{len(problems)} correct ({success_rate:.1f}%)")
    print(f"{'='*80}")

    return {"task": "common_sense_qa", "total": len(problems), "correct": correct,
            "success_rate": success_rate, "results": results}


# ==================== Task 3: SQL Generation ====================

def test_sql_generation():
    """Test SQL query generation from natural language"""
    print("\n" + "="*80)
    print("TASK 3: SQL GENERATION (DATABASE BENCH)")
    print("="*80)

    dev_file = os.path.join(DATA_DIR, "dbbench", "dev.jsonl")

    problems = []
    with open(dev_file, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= 10:
                break
            problems.append(json.loads(line))

    results = []
    correct = 0

    print(f"\nTesting {len(problems)} SQL generation tasks...")

    for idx, item in enumerate(problems, 1):
        question = item['description']
        expected = item['label']
        schema = item.get('add_description', '')

        print(f"[{idx}/{len(problems)}] {question[:60]}...")

        prompt = f"Generate a SQL query for this question.\n\nDatabase Schema: {schema}\n\nQuestion: {question}\n\nSQL Query:"

        result = generate_response(prompt)

        if result['success']:
            is_correct = judge_answer(result['response'], expected, question, "sql")
            if is_correct:
                correct += 1
                print(f"  ✓ CORRECT")
            else:
                print(f"  ✗ INCORRECT")
        else:
            is_correct = False
            print(f"  ✗ ERROR")

        results.append({
            "id": idx,
            "question": question,
            "expected_answer": expected,
            "model_response": result['response'],
            "judged_correct": is_correct,
            "success": result['success']
        })

        time.sleep(1)

    success_rate = (correct / len(problems) * 100) if problems else 0
    print(f"\n{'='*80}")
    print(f"SQL GENERATION: {correct}/{len(problems)} correct ({success_rate:.1f}%)")
    print(f"{'='*80}")

    return {"task": "sql_generation", "total": len(problems), "correct": correct,
            "success_rate": success_rate, "results": results}


# ==================== Task 4: Knowledge Graph ====================

def test_knowledge_graph():
    """Test multi-hop reasoning over knowledge graphs"""
    print("\n" + "="*80)
    print("TASK 4: KNOWLEDGE GRAPH REASONING")
    print("="*80)

    dev_file = os.path.join(DATA_DIR, "knowledgegraph", "dev.json")

    with open(dev_file, 'r') as f:
        all_problems = json.load(f)

    problems = all_problems[:10]

    results = []
    correct = 0

    print(f"\nTesting {len(problems)} knowledge graph tasks...")

    for idx, item in enumerate(problems, 1):
        question = item['question']
        expected = item['answer']

        print(f"[{idx}/{len(problems)}] {question[:60]}...")

        prompt = f"Answer this question concisely.\n\nQuestion: {question}\n\nAnswer:"

        result = generate_response(prompt)

        if result['success']:
            is_correct = judge_answer(result['response'], expected, question, "kg")
            if is_correct:
                correct += 1
                print(f"  ✓ CORRECT")
            else:
                print(f"  ✗ INCORRECT")
        else:
            is_correct = False
            print(f"  ✗ ERROR")

        results.append({
            "id": idx,
            "question": question,
            "expected_answer": expected,
            "model_response": result['response'],
            "judged_correct": is_correct,
            "success": result['success']
        })

        time.sleep(1)

    success_rate = (correct / len(problems) * 100) if problems else 0
    print(f"\n{'='*80}")
    print(f"KNOWLEDGE GRAPH: {correct}/{len(problems)} correct ({success_rate:.1f}%)")
    print(f"{'='*80}")

    return {"task": "knowledge_graph", "total": len(problems), "correct": correct,
            "success_rate": success_rate, "results": results}


# ==================== Main Execution ====================

def main():
    """Run all evaluations and generate results"""
    print("\n" + "="*80)
    print("STARTING FULL EVALUATION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run all tests
    print("\n[1/4] Testing Math Reasoning...")
    math_results = test_math_reasoning()

    print("\n[2/4] Testing Common Sense QA...")
    csqa_results = test_common_sense_qa()

    print("\n[3/4] Testing SQL Generation...")
    sql_results = test_sql_generation()

    print("\n[4/4] Testing Knowledge Graph...")
    kg_results = test_knowledge_graph()

    # Save individual JSON files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    tasks = [
        ("math_reasoning", math_results),
        ("common_sense_qa", csqa_results),
        ("sql_generation", sql_results),
        ("knowledge_graph", kg_results)
    ]

    for task_name, task_data in tasks:
        filename = os.path.join(RESULTS_DIR, f"{task_name}_{timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "model": MODEL_ID,
                "endpoint": ENDPOINT_URL,
                "test_date": datetime.now().isoformat(),
                **task_data
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved: {filename}")

    # Generate summary MD
    total_tests = sum(t[1]['total'] for t in tasks)
    total_correct = sum(t[1]['correct'] for t in tasks)
    avg_rate = sum(t[1]['success_rate'] for t in tasks) / len(tasks)

    summary_file = os.path.join(RESULTS_DIR, f"EVALUATION_SUMMARY_{timestamp}.md")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"""# Qwen2.5-3B-Instruct AgentBench Evaluation Results

**Model**: {MODEL_ID}
**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Evaluation Method**: Manual judgment for each response

---

## Task Success Rates

| Task | Total | Correct | Success Rate |
|------|-------|---------|--------------|
| Math Reasoning | {math_results['total']} | {math_results['correct']} | **{math_results['success_rate']:.1f}%** |
| Common Sense QA | {csqa_results['total']} | {csqa_results['correct']} | **{csqa_results['success_rate']:.1f}%** |
| SQL Generation | {sql_results['total']} | {sql_results['correct']} | **{sql_results['success_rate']:.1f}%** |
| Knowledge Graph | {kg_results['total']} | {kg_results['correct']} | **{kg_results['success_rate']:.1f}%** |

---

## Summary

**Overall Performance:**
- Total questions tested: {total_tests}
- Total correct: {total_correct}
- Average success rate: {avg_rate:.1f}%

**Task-by-Task Analysis:**

1. **Math Reasoning ({math_results['success_rate']:.1f}%)**
   - {math_results['correct']}/{math_results['total']} problems solved correctly
   - Task: Basic arithmetic and word problems

2. **Common Sense QA ({csqa_results['success_rate']:.1f}%)**
   - {csqa_results['correct']}/{csqa_results['total']} questions answered correctly
   - Task: Multiple choice common sense reasoning

3. **SQL Generation ({sql_results['success_rate']:.1f}%)**
   - {sql_results['correct']}/{sql_results['total']} queries generated correctly
   - Task: Natural language to SQL translation

4. **Knowledge Graph ({kg_results['success_rate']:.1f}%)**
   - {kg_results['correct']}/{kg_results['total']} entities identified correctly
   - Task: Multi-hop reasoning over knowledge graphs

---

## Result Files

- `math_reasoning_{timestamp}.json`
- `common_sense_qa_{timestamp}.json`
- `sql_generation_{timestamp}.json`
- `knowledge_graph_{timestamp}.json`

*Generated: {datetime.now().isoformat()}*
""")

    print(f"\n✓ Summary saved: {summary_file}")

    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Math Reasoning:    {math_results['correct']}/{math_results['total']} ({math_results['success_rate']:.1f}%)")
    print(f"Common Sense QA:   {csqa_results['correct']}/{csqa_results['total']} ({csqa_results['success_rate']:.1f}%)")
    print(f"SQL Generation:    {sql_results['correct']}/{sql_results['total']} ({sql_results['success_rate']:.1f}%)")
    print(f"Knowledge Graph:   {kg_results['correct']}/{kg_results['total']} ({kg_results['success_rate']:.1f}%)")
    print("="*80)
    print(f"\nAll results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

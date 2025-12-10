"""
Berkeley Function Calling Leaderboard (BFCL) Evaluation Script
Evaluates Qwen agent/function calling capabilities using BFCL test data
"""

import os
import json
import time
import requests
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configuration - Token from environment variable
HF_TOKEN = os.environ.get('HF_TOKEN', '')
if not HF_TOKEN:
    print("Error: Please set HF_TOKEN environment variable")
    print("Usage: export HF_TOKEN='your_token_here' && python3 berkeley_evaluation.py")
    exit(1)

# Your HuggingFace Inference Endpoint
ENDPOINT_URL = "https://mzkzbztaqbgxlted.us-east-1.aws.endpoints.huggingface.cloud"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
RESULTS_DIR = "/Users/mac/Documents/GitHub/Agent-benchmark-test/Results/Berkeley"

# BFCL data path
BFCL_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Berkeley/venv/lib/python3.12/site-packages/bfcl_eval/data"
)

print(f"\n{'='*80}")
print("BERKELEY FUNCTION CALLING LEADERBOARD EVALUATION")
print(f"Model: {MODEL_ID}")
print(f"{'='*80}\n")


# ==================== Helper Functions ====================

def generate_response(prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """Generate response using HuggingFace Inference Endpoint"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_p": 0.95,
            "return_full_text": False
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
                time.sleep(3)
            else:
                return {"success": False, "response": "", "error": str(e)}

    return {"success": False, "response": "", "error": "Max retries exceeded"}


def load_bfcl_data(test_name: str, limit: int = None) -> List[Dict]:
    """Load BFCL test data - loads ALL data if limit is None"""
    data_file = os.path.join(BFCL_DATA_PATH, f"BFCL_v4_{test_name}.json")
    if not os.path.exists(data_file):
        print(f"Warning: Data file not found: {data_file}")
        return []

    data = []
    with open(data_file, 'r') as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            data.append(json.loads(line))
    return data


def load_bfcl_answers(test_name: str) -> Dict[str, Any]:
    """Load BFCL ground truth answers"""
    answer_file = os.path.join(BFCL_DATA_PATH, "possible_answer", f"BFCL_v4_{test_name}.json")
    if not os.path.exists(answer_file):
        return {}

    answers = {}
    with open(answer_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            answers[item['id']] = item['ground_truth']
    return answers


def format_function_schema(functions: List[Dict]) -> str:
    """Format function definitions for the prompt"""
    formatted = []
    for func in functions:
        func_str = f"Function: {func['name']}\n"
        func_str += f"Description: {func.get('description', 'No description')}\n"
        func_str += f"Parameters: {json.dumps(func.get('parameters', {}), indent=2)}"
        formatted.append(func_str)
    return "\n\n".join(formatted)


def parse_function_call(response: str) -> Optional[Dict[str, Any]]:
    """Parse function call from model response"""
    # Pattern 1: JSON format {"name": "func", "arguments": {...}}
    json_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*\})\s*\}'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        try:
            return {
                "name": match.group(1),
                "arguments": json.loads(match.group(2))
            }
        except:
            pass

    # Pattern 2: function(arg1=val1, arg2=val2)
    func_pattern = r'(\w+(?:\.\w+)*)\s*\(\s*([^)]*)\s*\)'
    match = re.search(func_pattern, response)
    if match:
        func_name = match.group(1)
        args_str = match.group(2)
        args = {}

        # Parse keyword arguments
        arg_pattern = r'(\w+)\s*=\s*([^,]+)'
        for arg_match in re.finditer(arg_pattern, args_str):
            key = arg_match.group(1)
            value = arg_match.group(2).strip().strip('"\'')
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    pass
            args[key] = value

        return {"name": func_name, "arguments": args}

    return None


def evaluate_function_call(parsed_call: Optional[Dict], ground_truth: List[Dict]) -> bool:
    """Evaluate if the parsed function call matches ground truth"""
    if not parsed_call or not ground_truth:
        return False

    for gt in ground_truth:
        for func_name, expected_args in gt.items():
            if parsed_call['name'] == func_name or parsed_call['name'].endswith(func_name):
                all_args_match = True
                for arg_name, possible_values in expected_args.items():
                    if arg_name in parsed_call['arguments']:
                        actual_value = parsed_call['arguments'][arg_name]
                        if actual_value not in possible_values and str(actual_value) not in [str(v) for v in possible_values]:
                            all_args_match = False
                            break
                    elif "" not in possible_values:
                        all_args_match = False
                        break
                if all_args_match:
                    return True
    return False


# ==================== Test Categories ====================

def test_simple_function_calling(test_name: str = "simple_python", limit: int = 10):
    """Test simple function calling capability"""
    print(f"\n{'='*80}")
    print(f"TASK: SIMPLE FUNCTION CALLING ({test_name})")
    print(f"{'='*80}")

    data = load_bfcl_data(test_name, limit)
    answers = load_bfcl_answers(test_name)

    if not data:
        print(f"No test data found for {test_name}")
        return {"task": test_name, "total": 0, "correct": 0, "success_rate": 0, "results": []}

    results = []
    correct = 0

    print(f"\nTesting {len(data)} function calling tasks...")

    for idx, item in enumerate(data, 1):
        test_id = item['id']
        question = item['question'][0][0]['content']
        functions = item['function']
        ground_truth = answers.get(test_id, [])

        print(f"[{idx}/{len(data)}] {question[:60]}...")

        func_schema = format_function_schema(functions)
        prompt = f"""You are a helpful assistant that can call functions.

Available Functions:
{func_schema}

User Query: {question}

Respond with ONLY the function call in this format:
function_name(arg1=value1, arg2=value2)

Response:"""

        result = generate_response(prompt)

        if result['success']:
            parsed_call = parse_function_call(result['response'])
            is_correct = evaluate_function_call(parsed_call, ground_truth)

            if is_correct:
                correct += 1
                print(f"  ✓ CORRECT")
            else:
                print(f"  ✗ INCORRECT")
                if parsed_call:
                    print(f"    Got: {parsed_call['name']}({parsed_call['arguments']})")
        else:
            is_correct = False
            parsed_call = None
            print(f"  ✗ ERROR: {result['error'][:50]}")

        results.append({
            "id": test_id,
            "question": question,
            "model_response": result['response'],
            "parsed_call": parsed_call,
            "ground_truth": ground_truth,
            "judged_correct": is_correct,
            "success": result['success']
        })

        time.sleep(0.3)

    success_rate = (correct / len(data) * 100) if data else 0
    print(f"\n{'='*80}")
    print(f"{test_name.upper()}: {correct}/{len(data)} correct ({success_rate:.1f}%)")
    print(f"{'='*80}")

    return {"task": test_name, "total": len(data), "correct": correct,
            "success_rate": success_rate, "results": results}


def test_multiple_function_calling(limit: int = 10):
    return test_simple_function_calling("multiple", limit)


def test_parallel_function_calling(limit: int = 10):
    print(f"\n{'='*80}")
    print("TASK: PARALLEL FUNCTION CALLING")
    print(f"{'='*80}")

    data = load_bfcl_data("parallel", limit)
    answers = load_bfcl_answers("parallel")

    if not data:
        return {"task": "parallel", "total": 0, "correct": 0, "success_rate": 0, "results": []}

    results = []
    correct = 0

    print(f"\nTesting {len(data)} parallel function calling tasks...")

    for idx, item in enumerate(data, 1):
        test_id = item['id']
        question = item['question'][0][0]['content']
        functions = item['function']
        ground_truth = answers.get(test_id, [])

        print(f"[{idx}/{len(data)}] {question[:60]}...")

        func_schema = format_function_schema(functions)
        prompt = f"""You are a helpful assistant. Call MULTIPLE functions if needed.

Available Functions:
{func_schema}

User Query: {question}

Respond with function calls, one per line:
function_name(arg1=value1, arg2=value2)

Response:"""

        result = generate_response(prompt)

        if result['success']:
            parsed_call = parse_function_call(result['response'])
            is_correct = evaluate_function_call(parsed_call, ground_truth)

            if is_correct:
                correct += 1
                print(f"  ✓ CORRECT")
            else:
                print(f"  ✗ INCORRECT")
        else:
            is_correct = False
            parsed_call = None
            print(f"  ✗ ERROR")

        results.append({
            "id": test_id,
            "question": question,
            "model_response": result['response'],
            "parsed_call": parsed_call,
            "ground_truth": ground_truth,
            "judged_correct": is_correct,
            "success": result['success']
        })

        time.sleep(0.3)

    success_rate = (correct / len(data) * 100) if data else 0
    print(f"\n{'='*80}")
    print(f"PARALLEL: {correct}/{len(data)} correct ({success_rate:.1f}%)")
    print(f"{'='*80}")

    return {"task": "parallel", "total": len(data), "correct": correct,
            "success_rate": success_rate, "results": results}


def test_irrelevance_detection(limit: int = 10):
    print(f"\n{'='*80}")
    print("TASK: IRRELEVANCE DETECTION")
    print(f"{'='*80}")

    data = load_bfcl_data("irrelevance", limit)

    if not data:
        return {"task": "irrelevance", "total": 0, "correct": 0, "success_rate": 0, "results": []}

    results = []
    correct = 0

    print(f"\nTesting {len(data)} irrelevance detection tasks...")

    for idx, item in enumerate(data, 1):
        test_id = item['id']
        question = item['question'][0][0]['content']
        functions = item['function']

        print(f"[{idx}/{len(data)}] {question[:60]}...")

        func_schema = format_function_schema(functions)
        prompt = f"""You are a helpful assistant.
If none of the functions can answer the query, say "NO_FUNCTION_NEEDED".

Available Functions:
{func_schema}

User Query: {question}

If applicable: function_name(args)
If not applicable: NO_FUNCTION_NEEDED

Response:"""

        result = generate_response(prompt)

        if result['success']:
            response_lower = result['response'].lower()
            is_correct = (
                "no_function" in response_lower or
                "none" in response_lower or
                "cannot" in response_lower or
                "not applicable" in response_lower
            )

            if is_correct:
                correct += 1
                print(f"  ✓ CORRECT")
            else:
                print(f"  ✗ INCORRECT")
        else:
            is_correct = False
            print(f"  ✗ ERROR")

        results.append({
            "id": test_id,
            "question": question,
            "model_response": result['response'],
            "judged_correct": is_correct,
            "success": result['success']
        })

        time.sleep(0.3)

    success_rate = (correct / len(data) * 100) if data else 0
    print(f"\n{'='*80}")
    print(f"IRRELEVANCE: {correct}/{len(data)} correct ({success_rate:.1f}%)")
    print(f"{'='*80}")

    return {"task": "irrelevance", "total": len(data), "correct": correct,
            "success_rate": success_rate, "results": results}


# ==================== Generic Test Function ====================

def test_generic(test_name: str, limit: int = None, is_irrelevance: bool = False):
    """Generic test function for any BFCL category - tests ALL data if limit is None"""
    print(f"\n{'='*80}")
    print(f"TASK: {test_name.upper().replace('_', ' ')}")
    print(f"{'='*80}")

    data = load_bfcl_data(test_name, limit)
    answers = load_bfcl_answers(test_name)

    if not data:
        print(f"No test data found for {test_name}")
        return {"task": test_name, "total": 0, "correct": 0, "success_rate": 0, "results": []}

    results = []
    correct = 0

    print(f"\nTesting {len(data)} tasks...")

    for idx, item in enumerate(data, 1):
        test_id = item['id']
        question = item['question'][0][0]['content']
        functions = item['function']
        ground_truth = answers.get(test_id, [])

        print(f"[{idx}/{len(data)}] {question[:55]}...")

        func_schema = format_function_schema(functions)

        if is_irrelevance:
            prompt = f"""You are a helpful assistant.
If none of the functions can answer the query, say "NO_FUNCTION_NEEDED".

Available Functions:
{func_schema}

User Query: {question}

If applicable: function_name(args)
If not applicable: NO_FUNCTION_NEEDED

Response:"""
        else:
            prompt = f"""You are a helpful assistant that can call functions.

Available Functions:
{func_schema}

User Query: {question}

Respond with ONLY the function call in this format:
function_name(arg1=value1, arg2=value2)

Response:"""

        result = generate_response(prompt)

        if result['success']:
            if is_irrelevance:
                response_lower = result['response'].lower()
                is_correct = (
                    "no_function" in response_lower or
                    "none" in response_lower or
                    "cannot" in response_lower or
                    "not applicable" in response_lower
                )
            else:
                parsed_call = parse_function_call(result['response'])
                is_correct = evaluate_function_call(parsed_call, ground_truth)

            if is_correct:
                correct += 1
                print(f"  ✓ CORRECT")
            else:
                print(f"  ✗ INCORRECT")
        else:
            is_correct = False
            print(f"  ✗ ERROR: {result['error'][:40] if result['error'] else 'Unknown'}")

        results.append({
            "id": test_id,
            "question": question,
            "model_response": result['response'],
            "ground_truth": ground_truth,
            "judged_correct": is_correct,
            "success": result['success']
        })

        time.sleep(0.3)

    success_rate = (correct / len(data) * 100) if data else 0
    print(f"\n{test_name.upper()}: {correct}/{len(data)} correct ({success_rate:.1f}%)")

    return {"task": test_name, "total": len(data), "correct": correct,
            "success_rate": success_rate, "results": results}


# ==================== Main ====================

# All BFCL test categories
ALL_TEST_CATEGORIES = [
    # Non-live single turn
    ("simple_python", False),
    ("simple_java", False),
    ("simple_javascript", False),
    ("multiple", False),
    ("parallel", False),
    ("parallel_multiple", False),
    ("irrelevance", True),
    # Live single turn
    ("live_simple", False),
    ("live_multiple", False),
    ("live_parallel", False),
    ("live_parallel_multiple", False),
    ("live_irrelevance", True),
    ("live_relevance", False),
]

def main():
    print("\n" + "="*80)
    print("BERKELEY FUNCTION CALLING LEADERBOARD - FULL EVALUATION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Categories: {len(ALL_TEST_CATEGORIES)}")
    print("="*80)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for i, (test_name, is_irrelevance) in enumerate(ALL_TEST_CATEGORIES, 1):
        print(f"\n[{i}/{len(ALL_TEST_CATEGORIES)}] Testing {test_name}...")
        result = test_generic(test_name, limit=None, is_irrelevance=is_irrelevance)  # Test ALL data
        all_results[test_name] = result

        # Save individual result
        filename = os.path.join(RESULTS_DIR, f"bfcl_{test_name}_{timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "model": MODEL_ID,
                "endpoint": ENDPOINT_URL,
                "test_date": datetime.now().isoformat(),
                **result
            }, f, indent=2, ensure_ascii=False)

    # Calculate totals
    total_tests = sum(r['total'] for r in all_results.values())
    total_correct = sum(r['correct'] for r in all_results.values())
    valid_rates = [r['success_rate'] for r in all_results.values() if r['total'] > 0]
    avg_rate = sum(valid_rates) / len(valid_rates) if valid_rates else 0

    # Generate summary
    summary_file = os.path.join(RESULTS_DIR, f"BFCL_FULL_SUMMARY_{timestamp}.md")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"""# BFCL Full Evaluation Results

**Model**: {MODEL_ID}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Categories Tested**: {len(ALL_TEST_CATEGORIES)}

## Results by Category

| Category | Task | Correct | Total | Rate |
|----------|------|---------|-------|------|
""")
        # Non-live
        f.write("| **Non-Live** | | | | |\n")
        for name in ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple", "irrelevance"]:
            if name in all_results:
                r = all_results[name]
                f.write(f"| | {name} | {r['correct']} | {r['total']} | {r['success_rate']:.1f}% |\n")

        # Live
        f.write("| **Live** | | | | |\n")
        for name in ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance"]:
            if name in all_results:
                r = all_results[name]
                f.write(f"| | {name} | {r['correct']} | {r['total']} | {r['success_rate']:.1f}% |\n")

        f.write(f"""
## Summary

| Metric | Value |
|--------|-------|
| Total Questions | {total_tests} |
| Total Correct | {total_correct} |
| Overall Accuracy | {(total_correct/total_tests*100) if total_tests > 0 else 0:.1f}% |
| Average Category Rate | {avg_rate:.1f}% |

## Category Breakdown

### Non-Live Tests (Synthetic)
""")
        non_live = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple", "irrelevance"]
        for name in non_live:
            if name in all_results:
                r = all_results[name]
                f.write(f"- **{name}**: {r['correct']}/{r['total']} ({r['success_rate']:.1f}%)\n")

        f.write("\n### Live Tests (Real-world)\n")
        live = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance"]
        for name in live:
            if name in all_results:
                r = all_results[name]
                f.write(f"- **{name}**: {r['correct']}/{r['total']} ({r['success_rate']:.1f}%)\n")

        f.write(f"\n*Generated: {datetime.now().isoformat()}*\n")

    print(f"\n✓ Summary saved: {summary_file}")

    # Final console output
    print("\n" + "="*80)
    print("BFCL FULL EVALUATION COMPLETE")
    print("="*80)
    print("\nNon-Live Tests:")
    for name in ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple", "irrelevance"]:
        if name in all_results:
            r = all_results[name]
            print(f"  {name:25s}: {r['correct']:2d}/{r['total']:2d} ({r['success_rate']:5.1f}%)")

    print("\nLive Tests:")
    for name in ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance"]:
        if name in all_results:
            r = all_results[name]
            print(f"  {name:25s}: {r['correct']:2d}/{r['total']:2d} ({r['success_rate']:5.1f}%)")

    print("="*80)
    print(f"Total: {total_correct}/{total_tests} ({(total_correct/total_tests*100) if total_tests > 0 else 0:.1f}%)")
    print(f"Average Category Rate: {avg_rate:.1f}%")
    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

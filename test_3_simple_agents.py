"""
Simplified AgentBench Testing Script
Tests 3 different agent tasks using HuggingFace Dedicated Endpoint for Qwen2.5-3B-Instruct
Results will be saved to the Results directory
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

# Configuration
HF_TOKEN = "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"
ENDPOINT_URL = "https://mzkzbztaqbgxlted.us-east-1.aws.endpoints.huggingface.cloud"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
RESULTS_DIR = "/Users/mac/Documents/GitHub/Agent-benchmark-test/Results"

print(f"\n{'='*80}")
print("Initializing HuggingFace Dedicated Endpoint...")
print(f"Endpoint: {ENDPOINT_URL}")
print(f"Model: {MODEL_ID}")
print(f"{'='*80}\n")

def generate_response(prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """Generate response using HuggingFace dedicated endpoint"""
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

            # Extract generated text from response
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                generated_text = result.get('generated_text', result.get('text', ''))
            else:
                generated_text = str(result)

            return {
                "success": True,
                "response": generated_text,
                "error": None
            }
        except Exception as e:
            import traceback
            error_msg = str(e) if str(e) else "Unknown error"
            full_traceback = traceback.format_exc()

            print(f"  Error: {error_msg}")
            print(f"  Full traceback: {full_traceback[:500]}")

            if "loading" in error_msg.lower() or "initializing" in error_msg.lower() or "503" in error_msg:
                wait_time = min(30 * (attempt + 1), 90)
                print(f"  Endpoint initializing... waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif attempt < max_retries - 1:
                print(f"  Retrying (attempt {attempt + 2}/{max_retries})...")
                time.sleep(5)
            else:
                return {
                    "success": False,
                    "response": "",
                    "error": f"{error_msg}\n{full_traceback[:200]}"
                }

    return {
        "success": False,
        "response": "",
        "error": "Max retries exceeded"
    }


# ==================== Task 1: Lateral Thinking Puzzle Agent ====================

def test_lateral_thinking_puzzles():
    """Test Agent 1: Lateral Thinking Puzzle Reasoning"""
    print("\n" + "="*80)
    print("AGENT 1: Lateral Thinking Puzzle Reasoning")
    print("="*80)

    # Load puzzles from AgentBench data
    try:
        df = pd.read_excel('AgentBench/data/lateralthinkingpuzzle/dev.xlsx')
        puzzles = df.head(5).to_dict('records')  # Test with 5 puzzles
    except Exception as e:
        print(f"Warning: Could not load AgentBench data: {e}")
        print("Using backup puzzles...")
        puzzles = [
            {
                "story": "A man walks into a bar and asks for a glass of water. The bartender pulls out a gun and points it at him. The man says 'Thank you' and leaves.",
                "answer": "The man had hiccups, and the bartender scared them away."
            },
            {
                "story": "A man is found dead in a field with an unopened package next to him. There are no footprints around.",
                "answer": "His parachute failed to open, and he fell from a plane."
            },
            {
                "story": "A man lives on the 10th floor. Every day he takes the elevator down to the lobby. But when he comes back, he only takes the elevator to the 7th floor and walks up the rest.",
                "answer": "He is short and can only reach the button for the 7th floor."
            },
            {
                "story": "A woman shoots her husband, then holds him underwater for five minutes. Shortly after, they both go out and enjoy a wonderful dinner together.",
                "answer": "She is a photographer. She shot him with a camera and developed the photo in water."
            },
            {
                "story": "A man is lying dead in a room. There is a puddle of water and broken glass on the floor.",
                "answer": "He is a goldfish. His bowl broke."
            }
        ]

    results = []
    successful = 0

    print(f"\nTesting {len(puzzles)} lateral thinking puzzles...")

    for idx, puzzle in enumerate(puzzles, 1):
        story = puzzle.get('story', '')
        expected = puzzle.get('answer', '')

        print(f"\n[{idx}/{len(puzzles)}] Puzzle: {story[:80]}...")

        prompt = f"""You are solving a lateral thinking puzzle. Read the story carefully and provide a logical explanation.

Story: {story}

Think step by step and provide a creative but logical explanation for what happened. Keep your answer concise (2-3 sentences).

Answer:"""

        result = generate_response(prompt)

        if result['success']:
            successful += 1
            print(f"  ✓ Got response")
            print(f"  Response: {result['response'][:150]}...")
        else:
            print(f"  ✗ Error: {result['error']}")

        results.append({
            "puzzle_id": idx,
            "story": story,
            "expected_answer": expected,
            "model_response": result['response'],
            "success": result['success'],
            "error": result['error']
        })

    success_rate = (successful / len(puzzles) * 100) if puzzles else 0

    print(f"\n{'='*80}")
    print(f"AGENT 1 RESULTS:")
    print(f"  Total puzzles: {len(puzzles)}")
    print(f"  Successful responses: {successful}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"{'='*80}")

    return {
        "agent_name": "Lateral Thinking Puzzle Agent",
        "task_type": "reasoning",
        "total_tests": len(puzzles),
        "successful_responses": successful,
        "success_rate": success_rate,
        "results": results
    }


# ==================== Task 2: Math Reasoning Agent ====================

def test_math_reasoning():
    """Test Agent 2: Math Word Problem Reasoning"""
    print("\n" + "="*80)
    print("AGENT 2: Math Word Problem Reasoning")
    print("="*80)

    problems = [
        {"problem": "Sarah has 15 apples. She gives 4 apples to her friend and buys 7 more. How many apples does Sarah have now?", "answer": 18},
        {"problem": "A train travels 120 kilometers in 2 hours. What is its average speed in kilometers per hour?", "answer": 60},
        {"problem": "Tom has 50 dollars. He spends 12 dollars on lunch, 8 dollars on a book, and 5 dollars on snacks. How much money does he have left?", "answer": 25},
        {"problem": "A rectangle has a length of 12 meters and a width of 5 meters. What is its area?", "answer": 60},
        {"problem": "Lisa read 8 books in January, 12 books in February, and 10 books in March. What is the average number of books she read per month?", "answer": 10}
    ]

    results = []
    successful = 0
    correct = 0

    print(f"\nTesting {len(problems)} math word problems...")

    for idx, item in enumerate(problems, 1):
        problem = item['problem']
        expected = item['answer']

        print(f"\n[{idx}/{len(problems)}] Problem: {problem}")

        prompt = f"""Solve this math word problem step by step:

Problem: {problem}

Provide your solution in this format:
1. Show your reasoning
2. Calculate the answer
3. State "Final Answer: [number]"

Solution:"""

        result = generate_response(prompt)

        is_correct = False
        extracted_answer = None

        if result['success']:
            successful += 1
            response = result['response']

            # Try to extract answer
            import re
            answer_patterns = [
                r'Final Answer:\s*([+-]?\d+\.?\d*)',
                r'answer is\s*([+-]?\d+\.?\d*)',
                r'=\s*([+-]?\d+\.?\d*)',
            ]

            for pattern in answer_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        extracted_answer = float(match.group(1))
                        break
                    except:
                        pass

            # If no pattern matched, try to find last number
            if extracted_answer is None:
                numbers = re.findall(r'([+-]?\d+\.?\d*)', response)
                if numbers:
                    try:
                        extracted_answer = float(numbers[-1])
                    except:
                        pass

            if extracted_answer is not None and abs(extracted_answer - expected) < 0.01:
                is_correct = True
                correct += 1
                print(f"  ✓ Correct! Answer: {extracted_answer}")
            else:
                print(f"  ✗ Wrong. Expected: {expected}, Got: {extracted_answer}")
        else:
            print(f"  ✗ Error: {result['error']}")

        results.append({
            "problem_id": idx,
            "problem": problem,
            "expected_answer": expected,
            "extracted_answer": extracted_answer,
            "model_response": result['response'],
            "is_correct": is_correct,
            "success": result['success'],
            "error": result['error']
        })

    success_rate = (successful / len(problems) * 100) if problems else 0
    accuracy = (correct / len(problems) * 100) if problems else 0

    print(f"\n{'='*80}")
    print(f"AGENT 2 RESULTS:")
    print(f"  Total problems: {len(problems)}")
    print(f"  Successful responses: {successful}")
    print(f"  Correct answers: {correct}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"{'='*80}")

    return {
        "agent_name": "Math Reasoning Agent",
        "task_type": "math_reasoning",
        "total_tests": len(problems),
        "successful_responses": successful,
        "correct_answers": correct,
        "success_rate": success_rate,
        "accuracy": accuracy,
        "results": results
    }


# ==================== Task 3: Common Sense QA Agent ====================

def test_common_sense_qa():
    """Test Agent 3: Common Sense Question Answering"""
    print("\n" + "="*80)
    print("AGENT 3: Common Sense Question Answering")
    print("="*80)

    questions = [
        {
            "question": "What happens to ice when it gets warm?",
            "options": ["A) It melts", "B) It gets harder", "C) It explodes", "D) Nothing happens"],
            "answer": "A",
            "explanation": "Ice melts into water when heated"
        },
        {
            "question": "Where would you typically find a fish?",
            "options": ["A) In a tree", "B) In water", "C) In the sky", "D) In a desert"],
            "answer": "B",
            "explanation": "Fish live in water"
        },
        {
            "question": "What do plants need to grow?",
            "options": ["A) Darkness and cold", "B) Sunlight and water", "C) Only air", "D) Only soil"],
            "answer": "B",
            "explanation": "Plants need sunlight, water, and nutrients to grow"
        },
        {
            "question": "What is the typical result of not sleeping for a long time?",
            "options": ["A) Feeling energized", "B) Feeling tired", "C) Growing taller", "D) Becoming invisible"],
            "answer": "B",
            "explanation": "Lack of sleep causes tiredness"
        },
        {
            "question": "If you drop a ball, what will happen?",
            "options": ["A) It will fly up", "B) It will fall down", "C) It will disappear", "D) It will freeze"],
            "answer": "B",
            "explanation": "Gravity causes objects to fall downward"
        }
    ]

    results = []
    successful = 0
    correct = 0

    print(f"\nTesting {len(questions)} common sense questions...")

    for idx, item in enumerate(questions, 1):
        question = item['question']
        options = item['options']
        expected = item['answer']

        print(f"\n[{idx}/{len(questions)}] Question: {question}")

        prompt = f"""Answer this common sense question by selecting the most logical option.

Question: {question}

Options:
{chr(10).join(options)}

Think about what makes sense in the real world and choose the best answer. Respond with just the letter (A, B, C, or D) and a brief explanation.

Answer:"""

        result = generate_response(prompt)

        is_correct = False
        extracted_answer = None

        if result['success']:
            successful += 1
            response = result['response'].strip()

            # Extract answer letter
            import re
            match = re.search(r'\b([A-D])\b', response)
            if match:
                extracted_answer = match.group(1).upper()
                if extracted_answer == expected.upper():
                    is_correct = True
                    correct += 1
                    print(f"  ✓ Correct! Answer: {extracted_answer}")
                else:
                    print(f"  ✗ Wrong. Expected: {expected}, Got: {extracted_answer}")
            else:
                print(f"  ✗ Could not extract answer from response")
        else:
            print(f"  ✗ Error: {result['error']}")

        results.append({
            "question_id": idx,
            "question": question,
            "options": options,
            "expected_answer": expected,
            "extracted_answer": extracted_answer,
            "model_response": result['response'],
            "is_correct": is_correct,
            "success": result['success'],
            "error": result['error']
        })

    success_rate = (successful / len(questions) * 100) if questions else 0
    accuracy = (correct / len(questions) * 100) if questions else 0

    print(f"\n{'='*80}")
    print(f"AGENT 3 RESULTS:")
    print(f"  Total questions: {len(questions)}")
    print(f"  Successful responses: {successful}")
    print(f"  Correct answers: {correct}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"{'='*80}")

    return {
        "agent_name": "Common Sense QA Agent",
        "task_type": "common_sense_qa",
        "total_tests": len(questions),
        "successful_responses": successful,
        "correct_answers": correct,
        "success_rate": success_rate,
        "accuracy": accuracy,
        "results": results
    }


# ==================== Main Function ====================

def main():
    """Run all 3 agent tests and save results"""
    print("\n" + "="*80)
    print("AGENTBENCH SIMPLIFIED TESTING")
    print("Model: Qwen/Qwen2.5-3B-Instruct (Dedicated Endpoint)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Run all tests
    agent1_results = test_lateral_thinking_puzzles()
    agent2_results = test_math_reasoning()
    agent3_results = test_common_sense_qa()

    # Compile final results
    final_results = {
        "test_info": {
            "model": MODEL_ID,
            "endpoint": ENDPOINT_URL,
            "test_date": datetime.now().isoformat(),
            "total_agents_tested": 3
        },
        "agents": [
            agent1_results,
            agent2_results,
            agent3_results
        ],
        "summary": {
            "agent_1_lateral_thinking": {
                "success_rate": agent1_results['success_rate']
            },
            "agent_2_math_reasoning": {
                "success_rate": agent2_results['success_rate'],
                "accuracy": agent2_results['accuracy']
            },
            "agent_3_common_sense_qa": {
                "success_rate": agent3_results['success_rate'],
                "accuracy": agent3_results['accuracy']
            }
        }
    }

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save 3 separate JSON files for each agent
    agent1_file = os.path.join(RESULTS_DIR, f"agent1_lateral_thinking_{timestamp}.json")
    with open(agent1_file, 'w', encoding='utf-8') as f:
        json.dump({
            "agent_name": "Lateral Thinking Puzzle Agent",
            "model": MODEL_ID,
            "endpoint": ENDPOINT_URL,
            "test_date": datetime.now().isoformat(),
            **agent1_results
        }, f, indent=2, ensure_ascii=False)

    agent2_file = os.path.join(RESULTS_DIR, f"agent2_math_reasoning_{timestamp}.json")
    with open(agent2_file, 'w', encoding='utf-8') as f:
        json.dump({
            "agent_name": "Math Reasoning Agent",
            "model": MODEL_ID,
            "endpoint": ENDPOINT_URL,
            "test_date": datetime.now().isoformat(),
            **agent2_results
        }, f, indent=2, ensure_ascii=False)

    agent3_file = os.path.join(RESULTS_DIR, f"agent3_common_sense_qa_{timestamp}.json")
    with open(agent3_file, 'w', encoding='utf-8') as f:
        json.dump({
            "agent_name": "Common Sense QA Agent",
            "model": MODEL_ID,
            "endpoint": ENDPOINT_URL,
            "test_date": datetime.now().isoformat(),
            **agent3_results
        }, f, indent=2, ensure_ascii=False)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Agent 1 (Lateral Thinking): {agent1_results['success_rate']:.1f}% success rate")
    print(f"Agent 2 (Math Reasoning): {agent2_results['accuracy']:.1f}% accuracy")
    print(f"Agent 3 (Common Sense QA): {agent3_results['accuracy']:.1f}% accuracy")
    print("\n" + "="*80)
    print("Results saved to:")
    print(f"  - {agent1_file}")
    print(f"  - {agent2_file}")
    print(f"  - {agent3_file}")
    print("="*80)


if __name__ == "__main__":
    main()

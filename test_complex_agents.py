"""
Complex AgentBench Testing Script
Tests real agent tasks from AgentBench that require multi-step reasoning
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any

# Configuration
HF_TOKEN = "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"
ENDPOINT_URL = "https://mzkzbztaqbgxlted.us-east-1.aws.endpoints.huggingface.cloud"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
RESULTS_DIR = "/Users/mac/Documents/GitHub/Agent-benchmark-test/Results"

print(f"\n{'='*80}")
print("COMPLEX AGENTBENCH TESTING")
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

            print(f"  Error: {error_msg[:200]}")

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


# ==================== Task 1: SQL Generation (Database Bench) ====================

def test_sql_generation():
    """Test Agent 1: SQL Query Generation from Natural Language"""
    print("\n" + "="*80)
    print("AGENT 1: SQL Query Generation (Database Bench)")
    print("="*80)

    problems = [
        {
            "question": "Who is the opponent on November 1?",
            "context": "Table: game_records\nColumns: Game (TEXT), Date (TEXT), Opponent (TEXT), Score (TEXT), Location (TEXT), Record (TEXT)\nSample row: 1, November 1, Toronto Huskies, 68-66, Maple Leaf Gardens, 1-0",
            "expected_answer": "Toronto Huskies",
            "difficulty": "easy"
        },
        {
            "question": "What is the average score of home games?",
            "context": "Table: game_records\nColumns: Game (TEXT), Date (TEXT), Opponent (TEXT), Score (TEXT), Location (TEXT), Record (TEXT)",
            "expected_answer": "SQL query required",
            "difficulty": "medium"
        },
        {
            "question": "How many games were played at home venues?",
            "context": "Table: game_records with columns for location data",
            "expected_answer": "Needs COUNT query",
            "difficulty": "medium"
        }
    ]

    results = []
    successful = 0
    correct = 0

    print(f"\nTesting {len(problems)} SQL generation tasks...")

    for idx, item in enumerate(problems, 1):
        question = item['question']
        context = item['context']
        expected = item['expected_answer']

        print(f"\n[{idx}/{len(problems)}] Question: {question}")
        print(f"  Difficulty: {item['difficulty']}")

        prompt = f"""You are a SQL query generator. Given a natural language question and database schema, generate a SQL query to answer the question.

Database Schema:
{context}

Question: {question}

Generate a SQL query to answer this question. Provide your answer in this format:
SQL Query: [your query here]

Response:"""

        result = generate_response(prompt)

        if result['success']:
            successful += 1
            response = result['response']

            # Check if response contains SQL keywords
            has_sql = any(keyword in response.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'JOIN'])

            if has_sql:
                print(f"  ✓ Generated SQL query")
                print(f"  Response: {response[:150]}...")
            else:
                print(f"  ⚠ Response may not contain valid SQL")
        else:
            print(f"  ✗ Error: {result['error'][:100]}")

        results.append({
            "problem_id": idx,
            "question": question,
            "context": context,
            "expected_answer": expected,
            "difficulty": item['difficulty'],
            "model_response": result['response'],
            "success": result['success'],
            "error": result['error']
        })

    success_rate = (successful / len(problems) * 100) if problems else 0

    print(f"\n{'='*80}")
    print(f"AGENT 1 RESULTS:")
    print(f"  Total problems: {len(problems)}")
    print(f"  Successful responses: {successful}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"{'='*80}")

    return {
        "agent_name": "SQL Generation Agent",
        "task_type": "sql_generation",
        "total_tests": len(problems),
        "successful_responses": successful,
        "success_rate": success_rate,
        "results": results
    }


# ==================== Task 2: Knowledge Graph Reasoning ====================

def test_knowledge_graph_reasoning():
    """Test Agent 2: Multi-step Knowledge Graph Reasoning"""
    print("\n" + "="*80)
    print("AGENT 2: Knowledge Graph Reasoning")
    print("="*80)

    problems = [
        {
            "question": "What position did Pat Connaughton play?",
            "context": "Pat Connaughton is an author and basketball player. Need to find his basketball position through knowledge graph relations.",
            "steps": ["Find Pat Connaughton", "Get his basketball positions", "Return the position"],
            "expected_answer": "Guard",
            "difficulty": "medium"
        },
        {
            "question": "Name the sensor type of a digital camera that has Bayer color filter and ISO 5000?",
            "context": "Multi-step reasoning: Find cameras with Bayer filter → Find cameras with ISO 5000 → Find intersection → Get sensor type",
            "steps": ["Find cameras with Bayer", "Find cameras with ISO 5000", "Intersect results", "Get sensor type"],
            "expected_answer": "Live MOS",
            "difficulty": "hard"
        },
        {
            "question": "What team did the coach of Chicago Bulls in 1997 previously coach?",
            "context": "Multi-hop reasoning through knowledge graph",
            "steps": ["Find Bulls coach in 1997", "Find previous teams coached"],
            "expected_answer": "Requires KG navigation",
            "difficulty": "hard"
        }
    ]

    results = []
    successful = 0

    print(f"\nTesting {len(problems)} knowledge graph reasoning tasks...")

    for idx, item in enumerate(problems, 1):
        question = item['question']
        context = item['context']
        expected = item['expected_answer']

        print(f"\n[{idx}/{len(problems)}] Question: {question}")
        print(f"  Difficulty: {item['difficulty']}")

        prompt = f"""You are reasoning over a knowledge graph. Break down the following question into logical steps and provide your reasoning.

Context: {context}

Question: {question}

Think step by step:
1. What entities do we need to find?
2. What relations do we need to traverse?
3. What is the final answer?

Provide your reasoning and final answer:"""

        result = generate_response(prompt)

        if result['success']:
            successful += 1
            response = result['response']
            print(f"  ✓ Got response")
            print(f"  Response: {response[:200]}...")
        else:
            print(f"  ✗ Error: {result['error'][:100]}")

        results.append({
            "problem_id": idx,
            "question": question,
            "context": context,
            "expected_steps": item['steps'],
            "expected_answer": expected,
            "difficulty": item['difficulty'],
            "model_response": result['response'],
            "success": result['success'],
            "error": result['error']
        })

    success_rate = (successful / len(problems) * 100) if problems else 0

    print(f"\n{'='*80}")
    print(f"AGENT 2 RESULTS:")
    print(f"  Total problems: {len(problems)}")
    print(f"  Successful responses: {successful}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"{'='*80}")

    return {
        "agent_name": "Knowledge Graph Reasoning Agent",
        "task_type": "kg_reasoning",
        "total_tests": len(problems),
        "successful_responses": successful,
        "success_rate": success_rate,
        "results": results
    }


# ==================== Task 3: OS Command Generation ====================

def test_os_command_generation():
    """Test Agent 3: Linux Command Generation for System Tasks"""
    print("\n" + "="*80)
    print("AGENT 3: OS Command Generation")
    print("="*80)

    problems = [
        {
            "question": "There's a program writing 'succeed' to /root/log every 1-3 seconds. Find the interval.",
            "context": "A background process is writing to a file periodically. Need to monitor and determine the interval.",
            "expected_commands": ["watch", "tail", "sleep", "diff"],
            "expected_answer": "1, 2, or 3 seconds",
            "difficulty": "hard"
        },
        {
            "question": "How many hidden files are in /home? (not including subdirectories)",
            "context": "Count files starting with '.' in the /home directory",
            "expected_commands": ["ls -a", "grep", "wc -l"],
            "expected_answer": "Integer count",
            "difficulty": "medium"
        },
        {
            "question": "There's a program writing to a file in /root every 1 second. What is the filename?",
            "context": "Multiple files exist in /root. Need to find which one is being modified.",
            "expected_commands": ["ls -lt", "watch", "stat"],
            "expected_answer": "Specific filename",
            "difficulty": "hard"
        },
        {
            "question": "Create a bash alias 'calc' that evaluates mathematical expressions. Example: calc 2 * (9 / 3) should output 6",
            "context": "Need to create a shell function or alias that can evaluate arithmetic",
            "expected_commands": ["alias", "bc", "expr", "function"],
            "expected_answer": "Bash command or function",
            "difficulty": "hard"
        }
    ]

    results = []
    successful = 0

    print(f"\nTesting {len(problems)} OS command generation tasks...")

    for idx, item in enumerate(problems, 1):
        question = item['question']
        context = item['context']
        expected = item['expected_answer']

        print(f"\n[{idx}/{len(problems)}] Task: {question}")
        print(f"  Difficulty: {item['difficulty']}")

        prompt = f"""You are a Linux system administrator. Solve the following problem by providing the appropriate bash commands.

Problem: {question}

Context: {context}

Provide your solution with:
1. Explanation of your approach
2. The specific bash command(s) to solve this
3. Expected output or result

Solution:"""

        result = generate_response(prompt)

        if result['success']:
            successful += 1
            response = result['response']

            # Check if response contains command indicators
            has_commands = any(cmd in response for cmd in ['ls', 'cat', 'grep', 'find', 'watch', 'tail', '$', '#!/bin/bash'])

            if has_commands:
                print(f"  ✓ Generated command solution")
            else:
                print(f"  ⚠ May not contain specific commands")

            print(f"  Response: {response[:200]}...")
        else:
            print(f"  ✗ Error: {result['error'][:100]}")

        results.append({
            "problem_id": idx,
            "question": question,
            "context": context,
            "expected_commands": item['expected_commands'],
            "expected_answer": expected,
            "difficulty": item['difficulty'],
            "model_response": result['response'],
            "success": result['success'],
            "error": result['error']
        })

    success_rate = (successful / len(problems) * 100) if problems else 0

    print(f"\n{'='*80}")
    print(f"AGENT 3 RESULTS:")
    print(f"  Total problems: {len(problems)}")
    print(f"  Successful responses: {successful}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"{'='*80}")

    return {
        "agent_name": "OS Command Generation Agent",
        "task_type": "os_interaction",
        "total_tests": len(problems),
        "successful_responses": successful,
        "success_rate": success_rate,
        "results": results
    }


# ==================== Main Function ====================

def main():
    """Run all 3 complex agent tests and save results"""
    print("\n" + "="*80)
    print("COMPLEX AGENTBENCH TESTING")
    print("Model: Qwen/Qwen2.5-3B-Instruct (GPU Dedicated Endpoint)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Run all tests
    agent1_results = test_sql_generation()
    agent2_results = test_knowledge_graph_reasoning()
    agent3_results = test_os_command_generation()

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save 3 separate JSON files for each agent
    agent1_file = os.path.join(RESULTS_DIR, f"complex_agent1_sql_{timestamp}.json")
    with open(agent1_file, 'w', encoding='utf-8') as f:
        json.dump({
            "agent_name": "SQL Generation Agent",
            "model": MODEL_ID,
            "endpoint": ENDPOINT_URL,
            "test_date": datetime.now().isoformat(),
            **agent1_results
        }, f, indent=2, ensure_ascii=False)

    agent2_file = os.path.join(RESULTS_DIR, f"complex_agent2_kg_{timestamp}.json")
    with open(agent2_file, 'w', encoding='utf-8') as f:
        json.dump({
            "agent_name": "Knowledge Graph Reasoning Agent",
            "model": MODEL_ID,
            "endpoint": ENDPOINT_URL,
            "test_date": datetime.now().isoformat(),
            **agent2_results
        }, f, indent=2, ensure_ascii=False)

    agent3_file = os.path.join(RESULTS_DIR, f"complex_agent3_os_{timestamp}.json")
    with open(agent3_file, 'w', encoding='utf-8') as f:
        json.dump({
            "agent_name": "OS Command Generation Agent",
            "model": MODEL_ID,
            "endpoint": ENDPOINT_URL,
            "test_date": datetime.now().isoformat(),
            **agent3_results
        }, f, indent=2, ensure_ascii=False)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - COMPLEX AGENT TASKS")
    print("="*80)
    print(f"Agent 1 (SQL Generation): {agent1_results['success_rate']:.1f}% success rate")
    print(f"Agent 2 (KG Reasoning): {agent2_results['success_rate']:.1f}% success rate")
    print(f"Agent 3 (OS Commands): {agent3_results['success_rate']:.1f}% success rate")
    print("\n" + "="*80)
    print("Results saved to:")
    print(f"  - {agent1_file}")
    print(f"  - {agent2_file}")
    print(f"  - {agent3_file}")
    print("="*80)
    print("\nNote: These are complex agent tasks requiring multi-step reasoning,")
    print("tool use, and problem-solving. Lower accuracy is expected compared to")
    print("simple arithmetic tasks.")


if __name__ == "__main__":
    main()

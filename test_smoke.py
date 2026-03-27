"""
Smoke test for Kramer — runs a small discovery cycle against a sample dataset
using the OpenClaw gateway for LLM calls.
"""
import asyncio
import os
import sys

# Route LLM through OpenClaw gateway
os.environ["LLM_BACKEND"] = "openai_compatible"
os.environ["LLM_BASE_URL"] = "http://127.0.0.1:18789/v1/chat/completions"
os.environ["CLAUDE_MODEL"] = "claude-sonnet-4-6"

# Need the gateway token
token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
if not token:
    # Try to read from openclaw config
    import subprocess
    try:
        result = subprocess.run(
            ["openclaw", "gateway", "token"], capture_output=True, text=True
        )
        token = result.stdout.strip()
        if token:
            os.environ["OPENCLAW_GATEWAY_TOKEN"] = token
    except Exception:
        pass

if not token:
    print("ERROR: Need OPENCLAW_GATEWAY_TOKEN. Run: openclaw gateway token")
    sys.exit(1)

# Set as ANTHROPIC_API_KEY for the openai_compatible backend auth
os.environ["ANTHROPIC_API_KEY"] = token

async def main():
    from src.world_model.graph import WorldModel
    from src.orchestrator.cycle_manager import Orchestrator
    from src.orchestrator.sub_objectives import ObjectiveTracker
    from src.utils.llm_client import get_llm_client

    print("=" * 60)
    print("KRAMER SMOKE TEST")
    print("=" * 60)

    # 1. Test LLM client
    print("\n1. Testing LLM client through OpenClaw gateway...")
    client = get_llm_client()
    try:
        response = client.create_message(
            model="claude-sonnet-4-6",
            max_tokens=100,
            temperature=0.5,
            messages=[{"role": "user", "content": "Say 'Kramer is alive!' and nothing else."}],
        )
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        print(f"   LLM response: {text.strip()}")
        print(f"   Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")
        print("   ✅ LLM client works!")
    except Exception as e:
        print(f"   ❌ LLM client failed: {e}")
        sys.exit(1)

    # 2. Test WorldModel
    print("\n2. Testing WorldModel...")
    wm = WorldModel()
    fid = wm.add_finding(text="Test finding", confidence=0.8, metadata={"test": True})
    hid = wm.add_hypothesis(text="Test hypothesis", confidence=0.7)
    print(f"   Added finding: {fid[:8]}")
    print(f"   Added hypothesis: {hid[:8]}")
    print(f"   Nodes: {wm.graph.number_of_nodes()}")
    print("   ✅ WorldModel works!")

    # 3. Test ObjectiveTracker (sub-objective decomposition)
    print("\n3. Testing ObjectiveTracker...")
    tracker = ObjectiveTracker(wm)
    subs = await tracker.decompose("What factors influence ice cream sales?", num_questions=3)
    print(f"   Decomposed into {len(subs)} sub-questions:")
    for i, so in enumerate(subs, 1):
        print(f"   {i}. {so.question}")
    print("   ✅ ObjectiveTracker works!")

    # 4. Test cost tracker
    print("\n4. Testing cost tracker...")
    from src.utils.cost_tracker import CostTracker
    cost = CostTracker.calculate_cost("claude-sonnet-4-6", 1000, 500)
    print(f"   Cost for 1000 in / 500 out on sonnet-4-6: ${cost:.4f}")
    cost2 = CostTracker.calculate_cost("some-unknown-model", 1000, 500)
    print(f"   Cost for unknown model (default pricing): ${cost2:.4f}")
    print("   ✅ Cost tracker works!")

    # 5. Test AnalysisCodebase
    print("\n5. Testing AnalysisCodebase...")
    from src.kramer.analysis_codebase import AnalysisCodebase
    codebase = AnalysisCodebase(dataset_path="data/sample.csv")
    print(f"   Version: {codebase.version}")
    print(f"   Current script: {'(empty)' if not codebase.current_script else f'{len(codebase.current_script)} chars'}")
    codebase.accept("print('hello')", score=0.5, objective="test")
    print(f"   After accept: version={codebase.version}, score={codebase.get_current_score()}")
    print("   ✅ AnalysisCodebase works!")

    # 6. Test PlanningMemory
    print("\n6. Testing PlanningMemory...")
    from src.orchestrator.planning_memory import PlanningMemory
    pm = PlanningMemory()
    ctx = pm.get_planning_context()
    print(f"   Empty context: '{ctx}'")
    print("   ✅ PlanningMemory works!")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED ✅")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

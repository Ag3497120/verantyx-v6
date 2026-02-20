# LLM integration layer for Verantyx
# LLM = 分解機 (decomposer) only — not a reasoner, not an oracle
from llm.contract import LLMContract, LLMCandidate, LLMVerifySpec
from llm.gates import GateResult, GateA, GateB, run_gates

"""Temp: validate the RAGAS judge path end-to-end on ONE question (free-tier friendly)."""
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.m4_eval import evaluate_ragas
from config import LLM_MODEL, HAS_LLM_KEY

print(f"judge model = {LLM_MODEL} | HAS_LLM_KEY = {HAS_LLM_KEY}", flush=True)

question = "Nhân viên chính thức được nghỉ phép năm bao nhiêu ngày?"
context = ("Trích từ nghi_phep_nam_v2024.md. Nhân viên chính thức được hưởng 12 ngày phép năm. "
           "Mỗi 5 năm thâm niên được cộng thêm 1 ngày phép.")
answer = "Nhân viên chính thức được nghỉ 12 ngày phép năm, cộng thêm 1 ngày cho mỗi 5 năm thâm niên."
ground_truth = "Nhân viên chính thức được 12 ngày phép năm, cộng thêm 1 ngày mỗi 5 năm thâm niên."

t0 = time.time()
res = evaluate_ragas([question], [answer], [[context]], [ground_truth])
print(f"\nelapsed {time.time()-t0:.1f}s")
print(json.dumps({k: v for k, v in res.items() if k != "per_question"}, ensure_ascii=False, indent=2))

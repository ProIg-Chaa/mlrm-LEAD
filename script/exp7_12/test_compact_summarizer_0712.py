from summarize_compact_talr_matrix_0712 import extract_mcq, pope_label

assert extract_mcq("Answer: (B)") == "B"
assert extract_mcq("first (A), final answer: C") == "C"
assert pope_label("Final answer: yes") == "YES"
print("synthetic evaluator tests passed")

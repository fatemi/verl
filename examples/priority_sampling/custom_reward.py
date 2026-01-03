"""
Custom reward function using Qwen Math Eval Toolkit.

This provides more robust math answer verification than the default:
- Symbolic equivalence via SymPy (e.g., 1/3 == 0.333...)
- Numerical tolerance (rel_tol=1e-4)
- Better LaTeX parsing and normalization
- Handles fractions, percentages, matrices, etc.

Returns:
    1.0 for correct answer
    0.0 for incorrect answer (including missing \boxed{})
"""

import sys
from pathlib import Path

# Add this directory to path so qwen_math_eval_toolkit can be imported
_current_dir = Path(__file__).parent.resolve()
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
from qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute reward score for math problems using Qwen toolkit.
    
    Args:
        data_source: The data source identifier (e.g., "math", "lighteval/MATH")
        solution_str: The model's response (already decoded, without prompt)
        ground_truth: The correct answer to compare against
        extra_info: Optional extra information (unused)
    
    Returns:
        1.0 if answer is correct, 0.0 otherwise
    """
    # Note: solution_str is the decoded model response (not the full conversation).
    # The reward manager decodes with skip_special_tokens=True, but some stop tokens
    # may still be present depending on tokenizer config. Clean them just in case.
    response = solution_str
    for stop_token in ["</s>", "<|im_end|>", "<|endoftext|>"]:
        if stop_token in response:
            response = response.split(stop_token)[0]
    response = response.strip()
    
    # Require boxed answer for valid math response
    # This encourages the model to use proper \boxed{} format
    if "boxed" not in response:
        return 0.0
    
    # Extract answer using Qwen toolkit (handles various formats)
    extracted_answer = qwen_extract_answer(response, data_name="math")
    
    # Compare using symbolic math equivalence
    # This handles: fractions, decimals, LaTeX expressions, etc.
    if qwen_math_equal(prediction=extracted_answer, reference=ground_truth):
        return 1.0
    
    return 0.0

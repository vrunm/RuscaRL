from typing import List, Dict, Any


def math_verify(model_response: str, parameters: Dict[str, Any]) -> bool:
    """
    Verify model response using mathematical verification logic, using both math.py and math_dapo.py methods
    Returns True if either verification method passes
    
    Args:
        model_response: Model's response
        parameters: Verification parameter dictionary, format: {"answer": "answer"}
        
    Returns:
        bool: Verification result, True indicates correct, False indicates incorrect
    """
    
    from verl.utils.reward_score.math_dapo import compute_score as compute_score_dapo
    from verl.utils.reward_score.math import compute_score as compute_score_math
    
    # Extract answer from dictionary
    answer = parameters.get("answer")
    if answer is None:
        raise ValueError("math_verify: parameters dict must contain 'answer' key")
    
    # Use math_dapo's strict boxed verification mode
    result_dapo = compute_score_dapo(model_response, answer, strict_box_verify=True)
    if isinstance(result_dapo, dict):
        acc_dapo = result_dapo.get("acc", False)
    else:
        acc_dapo = result_dapo > 0.5
    
    # If dapo verification passes, return True directly
    if acc_dapo:
        return True
    
    # If dapo verification fails, use math.py verification method
    score_math = compute_score_math(model_response, answer)
    return score_math > 0.5


def word_count_range(model_response: str, parameters: Dict[str, Any]) -> bool:
    """
    Verify if the word count of model response is within the specified range
    
    Args:
        model_response: Model's response
        parameters: Verification parameter dictionary, format: {"min_count": 10, "max_count": 100}
        
    Returns:
        bool: Verification result, True indicates word count is within range, False indicates out of range
        
    Raises:
        ValueError: When parameter format is incorrect
    """
    # Extract min_count and max_count from dictionary
    min_count = parameters.get("min_count")
    max_count = parameters.get("max_count")
    
    if min_count is None or max_count is None:
        raise ValueError("word_count_range: parameters dict must contain 'min_count' and 'max_count' keys")
    
    try:
        min_count = int(min_count)
        max_count = int(max_count)
    except (ValueError, TypeError):
        raise ValueError(f"word_count_range: min_count and max_count must be integers, got min_count={min_count}, max_count={max_count}")
    
    if min_count > max_count:
        raise ValueError(f"word_count_range: min_count ({min_count}) cannot be greater than max_count ({max_count})")
        
    word_count = len(model_response.split())
    
    result = min_count <= word_count <= max_count
    print(f"word_count_range: response has {word_count} words, range [{min_count}, {max_count}], result: {result}")
    return result


# Verification function registry for dynamic invocation
VERIFICATION_FUNCTIONS = {
    'math_verify': math_verify,
    'word_count_range': word_count_range,
}


def get_verification_function(function_name: str):
    """
    Get verification function by function name
    
    Args:
        function_name: Verification function name
        
    Returns:
        Verification function or None (if function does not exist)
    """
    return VERIFICATION_FUNCTIONS.get(function_name)
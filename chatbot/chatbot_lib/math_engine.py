import re

class MathEngine:
    def __init__(self):
        self.ops = {
            "+": lambda x, y: x + y,
            "-": lambda x, y: x - y,
            "*": lambda x, y: x * y,
            "x": lambda x, y: x * y,
            "/": lambda x, y: x / y if y != 0 else "Cannot divide by zero",
            "plus": lambda x, y: x + y,
            "minus": lambda x, y: x - y,
            "times": lambda x, y: x * y,
            "divided": lambda x, y: x / y if y != 0 else "Cannot divide by zero"
        }

    def solve(self, prompt):
        # Extract numbers
        numbers = re.findall(r"[-+]?\d*\.?\d+", prompt)
        if len(numbers) < 2:
            return "I need at least two numbers to do math!"
        
        nums = [float(n) for n in numbers]
        
        # Determine operator
        prompt_lower = prompt.lower()
        selected_op = None
        for op_name, op_func in self.ops.items():
            if op_name in prompt_lower:
                selected_op = op_func
                break
        
        if not selected_op:
            return "I'm not sure what operation you want me to do."
        
        try:
            result = nums[0]
            for next_num in nums[1:]:
                result = selected_op(result, next_num)
            
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return f"The result is {result}"
        except Exception as e:
            return f"Sorry, I couldn't calculate that: {e}"

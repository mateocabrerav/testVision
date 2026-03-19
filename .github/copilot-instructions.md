# GitHub Copilot Instructions

## Project Overview
This is a **person tracking system** that uses:
- **Hardware**: NVIDIA Jetson AGX Orin (edge computing platform)
- **Camera**: Intel RealSense D435i (depth camera with RGB and infrared sensors)
- **Purpose**: Real-time person detection, tracking, and depth perception

### Hardware-Specific Considerations
- Optimize code for ARM architecture (Jetson AGX Orin)
- Leverage GPU acceleration when possible (CUDA, TensorRT)
- Use pyrealsense2 library for D435i camera integration
- Handle depth data efficiently (point clouds, depth frames)
- Consider real-time performance constraints
- Manage power consumption on embedded device

### Key Technologies
- Computer Vision: OpenCV, depth processing
- Deep Learning: TensorFlow/PyTorch (optimized for Jetson)
- Camera SDK: Intel RealSense SDK (pyrealsense2)
- Person Detection: YOLO, MobileNet, or similar lightweight models
- Tracking Algorithms: DeepSORT, Kalman filters, or similar

## Language Requirements
- All code must be written in **English** (variable names, function names, comments, docstrings, etc.)
- Use English for all documentation and comments

## Core Principle: Minimal Code
- **Use the least amount of code possible**
- Prefer concise, elegant solutions over verbose implementations
- Eliminate unnecessary code, variables, and abstractions
- If something can be done in fewer lines without sacrificing readability, do it
- Avoid over-engineering - simple is better than complex
- **Only create files when absolutely necessary**
- **Avoid creating unnecessary configuration, helper, or utility files**

## Python Code Standards

### Code Style
- Follow PEP 8 style guide
- Use type hints for function parameters and return types
- Maximum line length: 88 characters (Black formatter standard)
- Use snake_case for functions and variables
- Use PascalCase for class names
- Use UPPER_CASE for constants

### Single Responsibility Principle
- Each function should do ONE thing and do it well
- Each class should have a single, well-defined purpose
- If a function does multiple things, split it into smaller functions
- Keep functions small (ideally under 20 lines)
- Keep classes focused and cohesive

### Function Design
- Functions should be small and focused
- Aim for functions with 1-5 parameters maximum
- Use descriptive function names that clearly indicate their purpose
- Each function should have a single level of abstraction
- Avoid deep nesting (maximum 2-3 levels)

### Code Organization
- One class per file (unless closely related helper classes)
- Group related functions into modules
- Use clear, descriptive module names
- Separate business logic from infrastructure code
- **Create new files only when they serve a clear, specific purpose**
- **Avoid premature file/module separation**

### Documentation
- Add docstrings to all public functions, classes, and modules
- Use Google or NumPy docstring format
- Include type information in docstrings
- Document exceptions that can be raised

### Best Practices
- Use list comprehensions for simple transformations
- Prefer composition over inheritance
- Use context managers (with statements) for resource management
- Handle exceptions appropriately (don't use bare except clauses)
- Use logging instead of print statements for debugging
- Avoid global variables
- Use constants for magic numbers and strings
- **Write minimal code - every line should have a purpose**
- **Remove redundant code and unnecessary abstractions**
- **Don't create files "just in case" - create them when needed**

### Testing
- All test files must be placed in the `test/` directory
- Name test files with `test_` prefix (e.g., `test_module_name.py`)
- Use pytest as the testing framework
- Each test function should test one specific behavior
- Use descriptive test names that explain what is being tested
- Follow the Arrange-Act-Assert pattern in tests
- Aim for high test coverage (minimum 80%)

### Code Examples

#### Good Function Example:
```python
def calculate_total_price(items: list[dict], tax_rate: float) -> float:
    """
    Calculate the total price including tax.
    
    Args:
        items: List of item dictionaries with 'price' keys
        tax_rate: Tax rate as decimal (e.g., 0.1 for 10%)
    
    Returns:
        Total price including tax
    """
    subtotal = sum(item['price'] for item in items)
    return subtotal * (1 + tax_rate)
```

#### Good Class Example:
```python
class OrderProcessor:
    """Process customer orders."""
    
    def __init__(self, payment_gateway: PaymentGateway):
        self._payment_gateway = payment_gateway
    
    def process(self, order: Order) -> ProcessResult:
        """Process a single order."""
        # Implementation
        pass
```

### Avoid
- Large monolithic functions (over 30 lines)
- Deep nesting (more than 3 levels)
- Multiple responsibilities in one function/class
- Unclear variable names (x, temp, data, etc.)
- Magic numbers without constants
- Mutable default arguments
- Global state

## Project Structure
```
project_root/
├── src/
│   └── module_name/
│       ├── __init__.py
│       └── feature.py
├── test/
│   └── test_feature.py
├── requirements.txt
└── README.md
```

## Additional Guidelines
- Prefer pure functions when possible (no side effects)
- Use dependency injection for better testability
- Keep code DRY (Don't Repeat Yourself)
- Make it work, make it right, make it fast (in that order)
- Readability counts - code is read more often than written
- **Minimize code - less code means fewer bugs and easier maintenance**
- **Question every line - if it doesn't add value, remove it**
- **Question every file - only create when there's a clear need**
- **Resist the urge to over-organize - start simple, refactor when needed**

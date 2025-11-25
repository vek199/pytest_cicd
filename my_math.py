"""
Comprehensive math module for learning pytest and CI/CD.

This module demonstrates:
- Type hints
- Exception handling
- Classes and OOP
- Decorators
- Context managers
- Static methods and class methods
- Property decorators
"""

import logging
import math
import time
from functools import wraps
from typing import Any, Callable, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type alias for numeric types
Number = Union[int, float]


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class MathError(Exception):
    """Base exception for math operations."""

    pass


class DivisionByZeroError(MathError):
    """Raised when division by zero is attempted."""

    pass


class NegativeNumberError(MathError):
    """Raised when a negative number is not allowed."""

    pass


class InvalidInputError(MathError):
    """Raised when input is invalid."""

    pass


class EmptySequenceError(MathError):
    """Raised when sequence is empty but shouldn't be."""

    pass


# ============================================================================
# DECORATORS
# ============================================================================


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug(f"{func.__name__} took {end - start:.6f} seconds")
        return result

    return wrapper


def validate_numeric_args(func: Callable) -> Callable:
    """Decorator to validate that all positional args are numeric."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        for i, arg in enumerate(args):
            if not isinstance(arg, (int, float)):
                raise InvalidInputError(
                    f"Argument {i} must be numeric, got {type(arg).__name__}"
                )
        return func(*args, **kwargs)

    return wrapper


def retry(max_attempts: int = 3, delay: float = 0.1) -> Callable:
    """Decorator factory for retrying failed operations."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


# ============================================================================
# BASIC FUNCTIONS
# ============================================================================


@validate_numeric_args
def add(a: Number, b: Number) -> Number:
    """Add two numbers."""
    return a + b


@validate_numeric_args
def subtract(a: Number, b: Number) -> Number:
    """Subtract b from a."""
    return a - b


@validate_numeric_args
def multiply(a: Number, b: Number) -> Number:
    """Multiply two numbers."""
    return a * b


@validate_numeric_args
def divide(a: Number, b: Number) -> float:
    """
    Divide a by b.

    Raises:
        DivisionByZeroError: If b is zero.
    """
    if b == 0:
        raise DivisionByZeroError("Cannot divide by zero")
    return a / b


@validate_numeric_args
def power(base: Number, exponent: Number) -> Number:
    """Raise base to the power of exponent."""
    return base**exponent


@validate_numeric_args
def modulo(a: Number, b: Number) -> Number:
    """Return remainder of a divided by b."""
    if b == 0:
        raise DivisionByZeroError("Cannot perform modulo with zero")
    return a % b


def factorial(n: int) -> int:
    """
    Calculate factorial of n.

    Raises:
        NegativeNumberError: If n is negative.
        InvalidInputError: If n is not an integer.
    """
    if not isinstance(n, int):
        raise InvalidInputError("Factorial requires an integer")
    if n < 0:
        raise NegativeNumberError("Factorial not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def fibonacci(n: int) -> List[int]:
    """
    Generate first n Fibonacci numbers.

    Raises:
        NegativeNumberError: If n is negative.
        InvalidInputError: If n is not an integer.
    """
    if not isinstance(n, int):
        raise InvalidInputError("Fibonacci requires an integer")
    if n < 0:
        raise NegativeNumberError("Cannot generate negative Fibonacci numbers")
    if n == 0:
        return []
    if n == 1:
        return [0]

    fib = [0, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if not isinstance(n, int):
        raise InvalidInputError("is_prime requires an integer")
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor using Euclidean algorithm."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise InvalidInputError("GCD requires integers")
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Calculate least common multiple."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise InvalidInputError("LCM requires integers")
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


# ============================================================================
# CALCULATOR CLASS
# ============================================================================


class Calculator:
    """
    A calculator class demonstrating OOP concepts.

    Features:
    - Memory functionality
    - History tracking
    - Chainable operations
    """

    def __init__(self, initial_value: Number = 0):
        """Initialize calculator with optional starting value."""
        self._value: Number = initial_value
        self._memory: Number = 0
        self._history: List[str] = []
        self._precision: int = 10

    @property
    def value(self) -> Number:
        """Current calculator value."""
        return self._value

    @value.setter
    def value(self, new_value: Number) -> None:
        """Set calculator value."""
        if not isinstance(new_value, (int, float)):
            raise InvalidInputError("Value must be numeric")
        self._value = new_value

    @property
    def memory(self) -> Number:
        """Value stored in memory."""
        return self._memory

    @property
    def history(self) -> List[str]:
        """List of operations performed."""
        return self._history.copy()

    @property
    def precision(self) -> int:
        """Decimal precision for operations."""
        return self._precision

    @precision.setter
    def precision(self, value: int) -> None:
        """Set decimal precision."""
        if not isinstance(value, int) or value < 0:
            raise InvalidInputError("Precision must be a non-negative integer")
        self._precision = value

    def _record(self, operation: str) -> None:
        """Record an operation in history."""
        self._history.append(f"{operation} = {self._value}")

    def add(self, x: Number) -> "Calculator":
        """Add x to current value. Returns self for chaining."""
        old_value = self._value
        self._value = add(self._value, x)
        self._record(f"{old_value} + {x}")
        return self

    def subtract(self, x: Number) -> "Calculator":
        """Subtract x from current value. Returns self for chaining."""
        old_value = self._value
        self._value = subtract(self._value, x)
        self._record(f"{old_value} - {x}")
        return self

    def multiply(self, x: Number) -> "Calculator":
        """Multiply current value by x. Returns self for chaining."""
        old_value = self._value
        self._value = multiply(self._value, x)
        self._record(f"{old_value} * {x}")
        return self

    def divide(self, x: Number) -> "Calculator":
        """Divide current value by x. Returns self for chaining."""
        old_value = self._value
        self._value = divide(self._value, x)
        self._record(f"{old_value} / {x}")
        return self

    def power(self, x: Number) -> "Calculator":
        """Raise current value to power x. Returns self for chaining."""
        old_value = self._value
        self._value = power(self._value, x)
        self._record(f"{old_value} ^ {x}")
        return self

    def sqrt(self) -> "Calculator":
        """Calculate square root of current value."""
        if self._value < 0:
            raise NegativeNumberError("Cannot calculate square root of negative number")
        old_value = self._value
        self._value = math.sqrt(self._value)
        self._record(f"sqrt({old_value})")
        return self

    def clear(self) -> "Calculator":
        """Reset calculator to zero."""
        self._value = 0
        self._record("clear")
        return self

    def clear_history(self) -> "Calculator":
        """Clear operation history."""
        self._history.clear()
        return self

    def memory_store(self) -> "Calculator":
        """Store current value in memory."""
        self._memory = self._value
        return self

    def memory_recall(self) -> "Calculator":
        """Set current value to memory value."""
        self._value = self._memory
        self._record(f"MR({self._memory})")
        return self

    def memory_add(self) -> "Calculator":
        """Add current value to memory."""
        self._memory += self._value
        return self

    def memory_clear(self) -> "Calculator":
        """Clear memory."""
        self._memory = 0
        return self

    def round_value(self) -> "Calculator":
        """Round current value to set precision."""
        self._value = round(self._value, self._precision)
        return self

    @staticmethod
    def from_expression(expression: str) -> "Calculator":
        """
        Create a calculator from a simple expression.
        Supports: +, -, *, /

        Example: Calculator.from_expression("5 + 3")
        """
        parts = expression.split()
        if len(parts) != 3:
            raise InvalidInputError("Expression must be 'a op b' format")

        try:
            a = float(parts[0])
            op = parts[1]
            b = float(parts[2])
        except ValueError:
            raise InvalidInputError("Invalid numbers in expression")

        calc = Calculator(a)

        operations = {
            "+": calc.add,
            "-": calc.subtract,
            "*": calc.multiply,
            "/": calc.divide,
        }

        if op not in operations:
            raise InvalidInputError(f"Unknown operator: {op}")

        operations[op](b)
        return calc

    def __repr__(self) -> str:
        return f"Calculator(value={self._value})"

    def __str__(self) -> str:
        return str(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Calculator):
            return self._value == other._value
        if isinstance(other, (int, float)):
            return self._value == other
        return NotImplemented

    def __add__(self, other: Number) -> "Calculator":
        return Calculator(self._value).add(other)

    def __sub__(self, other: Number) -> "Calculator":
        return Calculator(self._value).subtract(other)

    def __mul__(self, other: Number) -> "Calculator":
        return Calculator(self._value).multiply(other)

    def __truediv__(self, other: Number) -> "Calculator":
        return Calculator(self._value).divide(other)


# ============================================================================
# STATISTICS CLASS
# ============================================================================


class Statistics:
    """
    Statistics calculator for numerical data analysis.

    Demonstrates:
    - Working with sequences
    - Class methods and static methods
    - Property decorators with computation
    """

    def __init__(self, data: Optional[List[Number]] = None):
        """Initialize with optional data list."""
        self._data: List[Number] = []
        if data:
            self.data = data

    @property
    def data(self) -> List[Number]:
        """The dataset."""
        return self._data.copy()

    @data.setter
    def data(self, values: List[Number]) -> None:
        """Set the dataset with validation."""
        if not isinstance(values, list):
            raise InvalidInputError("Data must be a list")
        for v in values:
            if not isinstance(v, (int, float)):
                raise InvalidInputError("All data values must be numeric")
        self._data = values.copy()

    @property
    def count(self) -> int:
        """Number of data points."""
        return len(self._data)

    @property
    def sum(self) -> Number:
        """Sum of all data points."""
        self._ensure_not_empty()
        return sum(self._data)

    @property
    def mean(self) -> float:
        """Arithmetic mean of the data."""
        self._ensure_not_empty()
        return self.sum / self.count

    @property
    def median(self) -> Number:
        """Median value of the data."""
        self._ensure_not_empty()
        sorted_data = sorted(self._data)
        n = len(sorted_data)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_data[mid - 1] + sorted_data[mid]) / 2
        return sorted_data[mid]

    @property
    def mode(self) -> List[Number]:
        """Mode(s) of the data (most frequent values)."""
        self._ensure_not_empty()
        frequency: dict = {}
        for value in self._data:
            frequency[value] = frequency.get(value, 0) + 1
        max_freq = max(frequency.values())
        return [k for k, v in frequency.items() if v == max_freq]

    @property
    def min(self) -> Number:
        """Minimum value in the data."""
        self._ensure_not_empty()
        return min(self._data)

    @property
    def max(self) -> Number:
        """Maximum value in the data."""
        self._ensure_not_empty()
        return max(self._data)

    @property
    def range(self) -> Number:
        """Range of the data (max - min)."""
        return self.max - self.min

    @property
    def variance(self) -> float:
        """Population variance of the data."""
        self._ensure_not_empty()
        mean = self.mean
        return sum((x - mean) ** 2 for x in self._data) / self.count

    @property
    def std_dev(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.variance)

    @property
    def sample_variance(self) -> float:
        """Sample variance (Bessel's correction)."""
        if self.count < 2:
            raise EmptySequenceError("Sample variance requires at least 2 data points")
        mean = self.mean
        return sum((x - mean) ** 2 for x in self._data) / (self.count - 1)

    @property
    def sample_std_dev(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.sample_variance)

    def _ensure_not_empty(self) -> None:
        """Raise exception if data is empty."""
        if not self._data:
            raise EmptySequenceError("Cannot perform operation on empty dataset")

    def add_value(self, value: Number) -> "Statistics":
        """Add a value to the dataset."""
        if not isinstance(value, (int, float)):
            raise InvalidInputError("Value must be numeric")
        self._data.append(value)
        return self

    def remove_value(self, value: Number) -> "Statistics":
        """Remove first occurrence of value from dataset."""
        try:
            self._data.remove(value)
        except ValueError:
            raise InvalidInputError(f"Value {value} not in dataset")
        return self

    def clear(self) -> "Statistics":
        """Clear all data."""
        self._data.clear()
        return self

    def percentile(self, p: Number) -> float:
        """
        Calculate the p-th percentile of the data.

        Args:
            p: Percentile to compute (0-100)
        """
        self._ensure_not_empty()
        if not 0 <= p <= 100:
            raise InvalidInputError("Percentile must be between 0 and 100")

        sorted_data = sorted(self._data)
        k = (p / 100) * (len(sorted_data) - 1)
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return sorted_data[int(k)]

        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)

    def quartiles(self) -> tuple:
        """Return Q1, Q2 (median), Q3."""
        return (self.percentile(25), self.percentile(50), self.percentile(75))

    def iqr(self) -> float:
        """Interquartile range (Q3 - Q1)."""
        q1, _, q3 = self.quartiles()
        return q3 - q1

    def z_score(self, value: Number) -> float:
        """Calculate z-score for a given value."""
        self._ensure_not_empty()
        if self.std_dev == 0:
            raise MathError("Cannot calculate z-score when standard deviation is 0")
        return (value - self.mean) / self.std_dev

    def outliers(self, threshold: float = 1.5) -> List[Number]:
        """
        Find outliers using IQR method.

        Args:
            threshold: IQR multiplier for outlier detection (default 1.5)
        """
        self._ensure_not_empty()
        q1, _, q3 = self.quartiles()
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return [x for x in self._data if x < lower_bound or x > upper_bound]

    def summary(self) -> dict:
        """Return a dictionary with all basic statistics."""
        self._ensure_not_empty()
        return {
            "count": self.count,
            "sum": self.sum,
            "mean": self.mean,
            "median": self.median,
            "mode": self.mode,
            "min": self.min,
            "max": self.max,
            "range": self.range,
            "variance": self.variance,
            "std_dev": self.std_dev,
        }

    @staticmethod
    def correlation(x_data: List[Number], y_data: List[Number]) -> float:
        """
        Calculate Pearson correlation coefficient between two datasets.

        Args:
            x_data: First dataset
            y_data: Second dataset

        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(x_data) != len(y_data):
            raise InvalidInputError("Datasets must have same length")
        if len(x_data) < 2:
            raise EmptySequenceError("Need at least 2 data points")

        n = len(x_data)
        mean_x = sum(x_data) / n
        mean_y = sum(y_data) / n

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_data, y_data))

        sum_sq_x = sum((x - mean_x) ** 2 for x in x_data)
        sum_sq_y = sum((y - mean_y) ** 2 for y in y_data)

        denominator = math.sqrt(sum_sq_x * sum_sq_y)

        if denominator == 0:
            raise MathError("Cannot calculate correlation - zero variance")

        return numerator / denominator

    @classmethod
    def from_range(cls, start: int, end: int, step: int = 1) -> "Statistics":
        """Create Statistics instance from a range of numbers."""
        return cls(list(range(start, end, step)))

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return f"Statistics(data={self._data})"

    def __str__(self) -> str:
        if not self._data:
            return "Statistics(empty)"
        return (
            f"Statistics(n={self.count}, mean={self.mean:.2f}, std={self.std_dev:.2f})"
        )


# ============================================================================
# CONTEXT MANAGER
# ============================================================================


class MathContext:
    """
    Context manager for mathematical operations with automatic cleanup.

    Demonstrates context manager protocol (__enter__ and __exit__).
    Useful for operations that need setup/teardown.
    """

    def __init__(self, precision: int = 10, raise_on_error: bool = True):
        """
        Initialize math context.

        Args:
            precision: Decimal precision for rounding
            raise_on_error: Whether to raise exceptions or return None
        """
        self.precision = precision
        self.raise_on_error = raise_on_error
        self.operations_count = 0
        self.errors: List[str] = []
        self._previous_precision: Optional[int] = None

    def __enter__(self) -> "MathContext":
        """Enter context and setup."""
        logger.debug(f"Entering MathContext with precision={self.precision}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context and cleanup."""
        logger.debug(
            f"Exiting MathContext: {self.operations_count} operations, "
            f"{len(self.errors)} errors"
        )
        # Return True to suppress exceptions if raise_on_error is False
        if exc_type is not None and not self.raise_on_error:
            self.errors.append(str(exc_val))
            return True
        return False

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function within this context."""
        self.operations_count += 1
        try:
            result = func(*args, **kwargs)
            if isinstance(result, float):
                result = round(result, self.precision)
            return result
        except Exception as e:
            if self.raise_on_error:
                raise
            self.errors.append(f"{func.__name__}: {str(e)}")
            return None


# ============================================================================
# ASYNC OPERATIONS (for async testing demonstration)
# ============================================================================


async def async_add(a: Number, b: Number) -> Number:
    """Async version of add for demonstration."""
    import asyncio

    await asyncio.sleep(0.01)  # Simulate async operation
    return add(a, b)


async def async_factorial(n: int) -> int:
    """Async version of factorial."""
    import asyncio

    await asyncio.sleep(0.01)
    return factorial(n)


async def async_batch_add(pairs: List[tuple]) -> List[Number]:
    """Add multiple pairs concurrently."""
    import asyncio

    tasks = [async_add(a, b) for a, b in pairs]
    return await asyncio.gather(*tasks)

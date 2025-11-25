"""
Comprehensive pytest test suite for my_math module.

This file demonstrates:
- Basic test functions
- Test classes
- Parametrized tests
- Fixtures (see conftest.py)
- Markers (skip, xfail, slow, etc.)
- Exception testing
- Mocking and patching
- Async testing
- Property-based hints
- Coverage considerations
- Test organization patterns
"""

import math
import time
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from my_math import (  # Basic functions; Classes; Exceptions; Decorators; Async functions
    Calculator,
    DivisionByZeroError,
    EmptySequenceError,
    InvalidInputError,
    MathContext,
    MathError,
    NegativeNumberError,
    Statistics,
    add,
    async_add,
    async_batch_add,
    async_factorial,
    divide,
    factorial,
    fibonacci,
    gcd,
    is_prime,
    lcm,
    modulo,
    multiply,
    power,
    retry,
    subtract,
    timing_decorator,
    validate_numeric_args,
)

# ============================================================================
# BASIC FUNCTION TESTS
# ============================================================================


class TestBasicArithmetic:
    """Tests for basic arithmetic operations."""

    @pytest.mark.unit
    def test_add_positive_numbers(self):
        """Test addition of positive numbers."""
        assert add(1, 2) == 3
        assert add(100, 200) == 300

    @pytest.mark.unit
    def test_add_negative_numbers(self):
        """Test addition with negative numbers."""
        assert add(-1, 1) == 0
        assert add(-1, -1) == -2
        assert add(-5, -10) == -15

    @pytest.mark.unit
    def test_add_floats(self):
        """Test addition with floating point numbers."""
        assert add(1.5, 2.5) == 4.0
        assert add(0.1, 0.2) == pytest.approx(0.3)  # Floating point comparison

    @pytest.mark.unit
    def test_subtract_basic(self):
        """Test basic subtraction."""
        assert subtract(5, 3) == 2
        assert subtract(3, 5) == -2
        assert subtract(0, 0) == 0

    @pytest.mark.unit
    def test_multiply_basic(self):
        """Test basic multiplication."""
        assert multiply(3, 4) == 12
        assert multiply(-2, 3) == -6
        assert multiply(-2, -3) == 6
        assert multiply(0, 100) == 0

    @pytest.mark.unit
    def test_divide_basic(self):
        """Test basic division."""
        assert divide(10, 2) == 5
        assert divide(7, 2) == 3.5
        assert divide(-10, 2) == -5

    @pytest.mark.unit
    def test_divide_by_zero_raises_exception(self):
        """Test that division by zero raises DivisionByZeroError."""
        with pytest.raises(DivisionByZeroError) as excinfo:
            divide(10, 0)
        assert "Cannot divide by zero" in str(excinfo.value)

    @pytest.mark.unit
    def test_power_basic(self):
        """Test power function."""
        assert power(2, 3) == 8
        assert power(3, 2) == 9
        assert power(2, 0) == 1
        assert power(2, -1) == 0.5

    @pytest.mark.unit
    def test_modulo_basic(self):
        """Test modulo operation."""
        assert modulo(10, 3) == 1
        assert modulo(15, 5) == 0
        assert modulo(7, 2) == 1

    @pytest.mark.unit
    def test_modulo_by_zero_raises_exception(self):
        """Test that modulo by zero raises exception."""
        with pytest.raises(DivisionByZeroError):
            modulo(10, 0)


class TestFactorial:
    """Tests for factorial function."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 6),
            (4, 24),
            (5, 120),
            (10, 3628800),
        ],
    )
    def test_factorial_valid_inputs(self, n: int, expected: int):
        """Test factorial with valid inputs using parametrize."""
        assert factorial(n) == expected

    @pytest.mark.unit
    def test_factorial_negative_raises_exception(self):
        """Test that factorial of negative number raises exception."""
        with pytest.raises(NegativeNumberError) as excinfo:
            factorial(-1)
        assert "negative" in str(excinfo.value).lower()

    @pytest.mark.unit
    def test_factorial_float_raises_exception(self):
        """Test that factorial of float raises exception."""
        with pytest.raises(InvalidInputError):
            factorial(3.5)

    @pytest.mark.slow
    @pytest.mark.performance
    def test_factorial_large_number(self):
        """Test factorial with large number (slow test)."""
        result = factorial(100)
        assert result > 0
        # 100! is approximately 9.33 × 10^157
        assert len(str(result)) > 150


class TestFibonacci:
    """Tests for Fibonacci sequence generation."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, []),
            (1, [0]),
            (2, [0, 1]),
            (5, [0, 1, 1, 2, 3]),
            (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]),
        ],
    )
    def test_fibonacci_sequences(self, n: int, expected: List[int]):
        """Test Fibonacci sequence generation."""
        assert fibonacci(n) == expected

    @pytest.mark.unit
    def test_fibonacci_negative_raises_exception(self):
        """Test that negative input raises exception."""
        with pytest.raises(NegativeNumberError):
            fibonacci(-5)

    @pytest.mark.unit
    def test_fibonacci_float_raises_exception(self):
        """Test that float input raises exception."""
        with pytest.raises(InvalidInputError):
            fibonacci(5.5)

    @pytest.mark.edge_case
    def test_fibonacci_property_sum(self):
        """Test Fibonacci property: fib[n] = fib[n-1] + fib[n-2]."""
        fib = fibonacci(20)
        for i in range(2, len(fib)):
            assert fib[i] == fib[i - 1] + fib[i - 2]


class TestPrimeNumbers:
    """Tests for prime number checking."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "n,expected",
        [
            (2, True),
            (3, True),
            (4, False),
            (5, True),
            (6, False),
            (7, True),
            (11, True),
            (13, True),
            (15, False),
            (17, True),
            (97, True),
            (100, False),
        ],
    )
    def test_is_prime(self, n: int, expected: bool):
        """Test prime number detection."""
        assert is_prime(n) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("n", [0, 1, -1, -5])
    def test_is_prime_non_primes(self, n: int):
        """Test that small and negative numbers are not prime."""
        assert is_prime(n) is False

    @pytest.mark.unit
    def test_is_prime_float_raises_exception(self):
        """Test that float input raises exception."""
        with pytest.raises(InvalidInputError):
            is_prime(5.5)

    @pytest.mark.integration
    def test_primes_in_session_fixture(self, prime_numbers):
        """Test using the session-scoped prime_numbers fixture."""
        # Verify first few primes
        assert prime_numbers[:10] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        # All numbers in the list should be prime
        for p in prime_numbers[:50]:
            assert is_prime(p)


class TestGCDAndLCM:
    """Tests for GCD and LCM functions."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (12, 8, 4),
            (100, 10, 10),
            (17, 13, 1),
            (0, 5, 5),
            (5, 0, 5),
            (48, 18, 6),
            (-12, 8, 4),  # Handles negatives
        ],
    )
    def test_gcd(self, a: int, b: int, expected: int):
        """Test GCD calculation."""
        assert gcd(a, b) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (4, 6, 12),
            (3, 5, 15),
            (12, 15, 60),
            (0, 5, 0),
            (1, 1, 1),
        ],
    )
    def test_lcm(self, a: int, b: int, expected: int):
        """Test LCM calculation."""
        assert lcm(a, b) == expected

    @pytest.mark.edge_case
    def test_gcd_lcm_relationship(self):
        """Test that gcd(a,b) * lcm(a,b) = |a * b|."""
        a, b = 24, 36
        assert gcd(a, b) * lcm(a, b) == abs(a * b)


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================


class TestParametrizedArithmetic:
    """Demonstrating various parametrization patterns."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (1, 1, 2),
            (0, 0, 0),
            (-1, -1, -2),
            (100, -100, 0),
            (0.1, 0.2, pytest.approx(0.3)),
        ],
    )
    def test_add_parametrized(self, a, b, expected):
        """Test addition with multiple parameter sets."""
        assert add(a, b) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    @pytest.mark.parametrize("b", [1, 2, 5, 10, 100])
    def test_multiplication_commutative(self, a, b):
        """Test that multiplication is commutative (a*b = b*a)."""
        assert multiply(a, b) == multiply(b, a)

    @pytest.mark.unit
    @pytest.mark.parametrize("a,b", [(x, y) for x in range(1, 6) for y in range(1, 6)])
    def test_division_and_multiplication_inverse(self, a, b):
        """Test that multiply and divide are inverse operations."""
        result = divide(multiply(a, b), b)
        assert result == pytest.approx(a)

    # Using fixture with parametrize
    @pytest.mark.unit
    def test_add_with_fixture(self, add_test_cases):
        """Test addition using parametrized fixture from conftest."""
        a, b, expected = add_test_cases
        assert add(a, b) == expected


# ============================================================================
# CALCULATOR CLASS TESTS
# ============================================================================


class TestCalculator:
    """Tests for Calculator class."""

    @pytest.mark.unit
    def test_calculator_initialization(self):
        """Test calculator initializes correctly."""
        calc = Calculator()
        assert calc.value == 0

        calc2 = Calculator(10)
        assert calc2.value == 10

    @pytest.mark.unit
    def test_calculator_basic_operations(self, calculator):
        """Test basic calculator operations using fixture."""
        calculator.add(5)
        assert calculator.value == 5

        calculator.subtract(2)
        assert calculator.value == 3

        calculator.multiply(4)
        assert calculator.value == 12

        calculator.divide(3)
        assert calculator.value == 4

    @pytest.mark.unit
    def test_calculator_chaining(self, calculator):
        """Test method chaining."""
        result = calculator.add(10).multiply(2).subtract(5).divide(3)
        assert result.value == 5
        assert result is calculator  # Same object returned

    @pytest.mark.unit
    def test_calculator_history(self, calculator):
        """Test that history is recorded."""
        calculator.add(5).multiply(2)
        history = calculator.history
        assert len(history) == 2
        assert "0 + 5" in history[0]
        assert "5 * 2" in history[1]

    @pytest.mark.unit
    def test_calculator_memory_operations(self, calculator):
        """Test memory store and recall."""
        calculator.add(100)
        calculator.memory_store()
        assert calculator.memory == 100

        calculator.clear()
        assert calculator.value == 0

        calculator.memory_recall()
        assert calculator.value == 100

    @pytest.mark.unit
    def test_calculator_sqrt(self, calculator):
        """Test square root operation."""
        calculator.add(16).sqrt()
        assert calculator.value == 4

    @pytest.mark.unit
    def test_calculator_sqrt_negative_raises_exception(self, calculator):
        """Test sqrt of negative raises exception."""
        calculator.subtract(10)  # value = -10
        with pytest.raises(NegativeNumberError):
            calculator.sqrt()

    @pytest.mark.unit
    def test_calculator_division_by_zero(self, calculator):
        """Test division by zero raises exception."""
        calculator.add(10)
        with pytest.raises(DivisionByZeroError):
            calculator.divide(0)

    @pytest.mark.unit
    def test_calculator_equality(self):
        """Test calculator equality comparison."""
        calc1 = Calculator(10)
        calc2 = Calculator(10)
        calc3 = Calculator(20)

        assert calc1 == calc2
        assert calc1 != calc3
        assert calc1 == 10
        assert calc1 != 20

    @pytest.mark.unit
    def test_calculator_operators(self):
        """Test dunder methods for operators."""
        calc = Calculator(10)

        result = calc + 5
        assert result.value == 15

        result = calc - 3
        assert result.value == 7

        result = calc * 2
        assert result.value == 20

        result = calc / 2
        assert result.value == 5

    @pytest.mark.unit
    def test_calculator_from_expression(self):
        """Test creating calculator from expression."""
        calc = Calculator.from_expression("5 + 3")
        assert calc.value == 8

        calc = Calculator.from_expression("10 * 3")
        assert calc.value == 30

    @pytest.mark.unit
    def test_calculator_from_expression_invalid(self):
        """Test invalid expression raises exception."""
        with pytest.raises(InvalidInputError):
            Calculator.from_expression("5 +")  # Incomplete

        with pytest.raises(InvalidInputError):
            Calculator.from_expression("5 @ 3")  # Invalid operator

    @pytest.mark.unit
    def test_calculator_repr_and_str(self, calculator):
        """Test string representations."""
        calculator.add(42)
        assert repr(calculator) == "Calculator(value=42)"
        assert str(calculator) == "42"

    @pytest.mark.unit
    def test_calculator_precision(self, calculator):
        """Test precision setting and rounding."""
        calculator.precision = 2
        assert calculator.precision == 2

        calculator.add(1).divide(3)  # 0.333...
        calculator.round_value()
        assert calculator.value == 0.33

    @pytest.mark.unit
    def test_calculator_factory_fixture(self, calculator_factory):
        """Test using the factory fixture."""
        calc1 = calculator_factory(10)
        calc2 = calculator_factory(20)

        assert calc1.value == 10
        assert calc2.value == 20
        assert calc1 is not calc2


class TestCalculatorWithClassFixture:
    """Tests using class-scoped fixture."""

    @pytest.fixture(autouse=True)
    def setup(self, class_calculator):
        """Setup using class-scoped fixture."""
        # class_calculator is available on self.calculator via conftest
        pass

    def test_shared_calculator_initial(self):
        """Test initial value from class fixture."""
        assert self.calculator.value == 100

    def test_shared_calculator_after_operation(self):
        """Test that state persists within class."""
        self.calculator.add(50)
        assert self.calculator.value == 150


# ============================================================================
# STATISTICS CLASS TESTS
# ============================================================================


class TestStatistics:
    """Tests for Statistics class."""

    @pytest.mark.unit
    def test_statistics_initialization(self):
        """Test statistics initialization."""
        stats = Statistics()
        assert stats.count == 0

        stats2 = Statistics([1, 2, 3])
        assert stats2.count == 3

    @pytest.mark.unit
    def test_statistics_basic_measures(self, statistics_with_data):
        """Test basic statistical measures."""
        # Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert statistics_with_data.count == 10
        assert statistics_with_data.sum == 55
        assert statistics_with_data.mean == 5.5
        assert statistics_with_data.median == 5.5
        assert statistics_with_data.min == 1
        assert statistics_with_data.max == 10
        assert statistics_with_data.range == 9

    @pytest.mark.unit
    def test_statistics_variance_and_std(self, statistics_with_data):
        """Test variance and standard deviation."""
        # Population variance for [1..10]
        assert statistics_with_data.variance == pytest.approx(8.25)
        assert statistics_with_data.std_dev == pytest.approx(math.sqrt(8.25))

    @pytest.mark.unit
    def test_statistics_sample_variance(self, statistics_with_data):
        """Test sample variance (with Bessel's correction)."""
        # Sample variance = n/(n-1) * population variance
        expected = 8.25 * 10 / 9
        assert statistics_with_data.sample_variance == pytest.approx(expected)

    @pytest.mark.unit
    def test_statistics_mode(self):
        """Test mode calculation."""
        stats = Statistics([1, 2, 2, 3, 3, 3, 4])
        assert stats.mode == [3]

        # Multiple modes
        stats2 = Statistics([1, 1, 2, 2, 3])
        assert set(stats2.mode) == {1, 2}

    @pytest.mark.unit
    def test_statistics_percentile(self, statistics_with_data):
        """Test percentile calculation."""
        assert statistics_with_data.percentile(50) == pytest.approx(5.5)
        assert statistics_with_data.percentile(0) == pytest.approx(1)
        assert statistics_with_data.percentile(100) == pytest.approx(10)

    @pytest.mark.unit
    def test_statistics_quartiles(self, statistics_with_data):
        """Test quartile calculation."""
        q1, q2, q3 = statistics_with_data.quartiles()
        assert q2 == pytest.approx(5.5)  # Median

    @pytest.mark.unit
    def test_statistics_iqr(self, statistics_with_data):
        """Test interquartile range."""
        iqr = statistics_with_data.iqr()
        assert iqr > 0

    @pytest.mark.unit
    def test_statistics_z_score(self, statistics_with_data):
        """Test z-score calculation."""
        mean = statistics_with_data.mean
        std = statistics_with_data.std_dev

        # z-score of mean should be 0
        assert statistics_with_data.z_score(mean) == pytest.approx(0)

        # z-score of mean + std should be 1
        assert statistics_with_data.z_score(mean + std) == pytest.approx(1)

    @pytest.mark.unit
    def test_statistics_outliers(self):
        """Test outlier detection."""
        # Add obvious outliers
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]  # 100 is outlier
        stats = Statistics(data)
        outliers = stats.outliers()
        assert 100 in outliers

    @pytest.mark.unit
    def test_statistics_add_remove_value(self, statistics):
        """Test adding and removing values."""
        statistics.add_value(5)
        statistics.add_value(10)
        assert statistics.count == 2
        assert statistics.sum == 15

        statistics.remove_value(5)
        assert statistics.count == 1
        assert statistics.sum == 10

    @pytest.mark.unit
    def test_statistics_remove_nonexistent_raises_exception(self, statistics_with_data):
        """Test removing non-existent value raises exception."""
        with pytest.raises(InvalidInputError):
            statistics_with_data.remove_value(999)

    @pytest.mark.unit
    def test_statistics_empty_raises_exception(self, statistics):
        """Test that operations on empty data raise exception."""
        with pytest.raises(EmptySequenceError):
            _ = statistics.mean

        with pytest.raises(EmptySequenceError):
            _ = statistics.sum

    @pytest.mark.unit
    def test_statistics_correlation(self):
        """Test Pearson correlation coefficient."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]

        # Perfect positive correlation
        assert Statistics.correlation(x, y) == pytest.approx(1.0)

        # Perfect negative correlation
        y_neg = [10, 8, 6, 4, 2]
        assert Statistics.correlation(x, y_neg) == pytest.approx(-1.0)

    @pytest.mark.unit
    def test_statistics_correlation_unequal_length_raises(self):
        """Test correlation with unequal lengths raises exception."""
        with pytest.raises(InvalidInputError):
            Statistics.correlation([1, 2, 3], [1, 2])

    @pytest.mark.unit
    def test_statistics_from_range(self):
        """Test creating statistics from range."""
        stats = Statistics.from_range(1, 11)  # 1 to 10
        assert stats.count == 10
        assert stats.min == 1
        assert stats.max == 10

    @pytest.mark.unit
    def test_statistics_summary(self, statistics_with_data):
        """Test summary method."""
        summary = statistics_with_data.summary()

        assert "count" in summary
        assert "mean" in summary
        assert "median" in summary
        assert "std_dev" in summary
        assert summary["count"] == 10

    @pytest.mark.unit
    def test_statistics_mean_fixture(self, mean_test_cases):
        """Test mean using parametrized fixture."""
        data, expected_mean = mean_test_cases
        stats = Statistics(data)
        assert stats.mean == pytest.approx(expected_mean)


# ============================================================================
# CONTEXT MANAGER TESTS
# ============================================================================


class TestMathContext:
    """Tests for MathContext class."""

    @pytest.mark.unit
    def test_math_context_basic_usage(self, math_context):
        """Test basic context manager usage."""
        with math_context:
            result = math_context.execute(add, 1, 2)
            assert result == 3
            assert math_context.operations_count == 1

    @pytest.mark.unit
    def test_math_context_precision(self):
        """Test precision is applied to results."""
        with MathContext(precision=2) as ctx:
            result = ctx.execute(divide, 1, 3)
            assert result == 0.33

    @pytest.mark.unit
    def test_math_context_error_raising(self):
        """Test that errors are raised when raise_on_error=True."""
        with pytest.raises(DivisionByZeroError):
            with MathContext(raise_on_error=True) as ctx:
                ctx.execute(divide, 1, 0)

    @pytest.mark.unit
    def test_math_context_error_suppression(self, silent_math_context):
        """Test that errors are suppressed when raise_on_error=False."""
        with silent_math_context as ctx:
            result = ctx.execute(divide, 1, 0)
            assert result is None
            assert len(ctx.errors) == 1
            assert "divide" in ctx.errors[0]

    @pytest.mark.unit
    def test_math_context_multiple_operations(self, math_context):
        """Test multiple operations in context."""
        with math_context:
            math_context.execute(add, 1, 2)
            math_context.execute(multiply, 3, 4)
            math_context.execute(divide, 10, 2)

            assert math_context.operations_count == 3


# ============================================================================
# EXCEPTION TESTING
# ============================================================================


class TestExceptions:
    """Tests for exception handling."""

    @pytest.mark.unit
    def test_division_by_zero_exception(self):
        """Test DivisionByZeroError is raised correctly."""
        with pytest.raises(DivisionByZeroError) as excinfo:
            divide(10, 0)

        assert "zero" in str(excinfo.value).lower()
        assert excinfo.type is DivisionByZeroError

    @pytest.mark.unit
    def test_negative_number_exception(self):
        """Test NegativeNumberError is raised correctly."""
        with pytest.raises(NegativeNumberError):
            factorial(-5)

    @pytest.mark.unit
    def test_invalid_input_exception(self):
        """Test InvalidInputError is raised correctly."""
        with pytest.raises(InvalidInputError):
            add("not", "numbers")

    @pytest.mark.unit
    def test_empty_sequence_exception(self):
        """Test EmptySequenceError is raised correctly."""
        stats = Statistics()
        with pytest.raises(EmptySequenceError):
            _ = stats.mean

    @pytest.mark.unit
    def test_exception_inheritance(self):
        """Test that custom exceptions inherit from MathError."""
        assert issubclass(DivisionByZeroError, MathError)
        assert issubclass(NegativeNumberError, MathError)
        assert issubclass(InvalidInputError, MathError)
        assert issubclass(EmptySequenceError, MathError)

    @pytest.mark.unit
    def test_exception_catching_base_class(self):
        """Test that MathError catches all custom exceptions."""
        with pytest.raises(MathError):
            divide(1, 0)

        with pytest.raises(MathError):
            factorial(-1)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "func,args,expected_exception",
        [
            (divide, (1, 0), DivisionByZeroError),
            (modulo, (1, 0), DivisionByZeroError),
            (factorial, (-1,), NegativeNumberError),
            (fibonacci, (-1,), NegativeNumberError),
            (add, ("a", "b"), InvalidInputError),
            (factorial, (3.5,), InvalidInputError),
        ],
    )
    def test_various_exceptions(self, func, args, expected_exception):
        """Test various exception scenarios."""
        with pytest.raises(expected_exception):
            func(*args)


# ============================================================================
# MOCKING AND PATCHING TESTS
# ============================================================================


class TestMocking:
    """Tests demonstrating mocking and patching."""

    @pytest.mark.unit
    def test_mock_basic(self):
        """Test basic mock usage."""
        mock_calc = Mock(spec=Calculator)
        mock_calc.value = 42
        mock_calc.add.return_value = mock_calc

        mock_calc.add(10)
        mock_calc.add.assert_called_once_with(10)
        assert mock_calc.value == 42

    @pytest.mark.unit
    def test_patch_function(self):
        """Test patching a function."""
        with patch("my_math.add") as mock_add:
            mock_add.return_value = 999

            # Now add returns mocked value
            result = mock_add(1, 2)
            assert result == 999
            mock_add.assert_called_with(1, 2)

    @pytest.mark.unit
    def test_patch_decorator(self):
        """Test using patch as decorator."""
        with patch.object(Calculator, "add") as mock_add:
            mock_add.return_value = MagicMock()

            calc = Calculator()
            calc.add(10)
            mock_add.assert_called_once_with(10)

    @pytest.mark.unit
    def test_mock_side_effect(self):
        """Test mock with side_effect."""
        mock_func = Mock()
        mock_func.side_effect = [1, 2, 3]

        assert mock_func() == 1
        assert mock_func() == 2
        assert mock_func() == 3

    @pytest.mark.unit
    def test_mock_side_effect_exception(self):
        """Test mock that raises exception."""
        mock_func = Mock()
        mock_func.side_effect = DivisionByZeroError("mocked error")

        with pytest.raises(DivisionByZeroError):
            mock_func()

    @pytest.mark.unit
    def test_patch_time(self):
        """Test patching time for deterministic tests."""
        with patch("time.perf_counter") as mock_time:
            mock_time.side_effect = [0.0, 1.0]  # Start and end times

            start = time.perf_counter()
            # ... some operation ...
            end = time.perf_counter()

            assert end - start == 1.0

    @pytest.mark.unit
    def test_mock_call_count(self):
        """Test tracking call counts."""
        mock = Mock()

        mock()
        mock()
        mock()

        assert mock.call_count == 3

    @pytest.mark.unit
    def test_mock_call_args(self):
        """Test tracking call arguments."""
        mock = Mock()

        mock(1, 2, key="value")
        mock(3, 4, key="other")

        # Check last call
        mock.assert_called_with(3, 4, key="other")

        # Check all calls
        assert mock.call_args_list[0] == ((1, 2), {"key": "value"})
        assert mock.call_args_list[1] == ((3, 4), {"key": "other"})


# ============================================================================
# ASYNC TESTS
# ============================================================================


class TestAsyncFunctions:
    """Tests for async functions."""

    @pytest.mark.asyncio
    async def test_async_add(self):
        """Test async add function."""
        result = await async_add(5, 3)
        assert result == 8

    @pytest.mark.asyncio
    async def test_async_factorial(self):
        """Test async factorial function."""
        result = await async_factorial(5)
        assert result == 120

    @pytest.mark.asyncio
    async def test_async_batch_add(self):
        """Test batch async add."""
        pairs = [(1, 2), (3, 4), (5, 6)]
        results = await async_batch_add(pairs)
        assert results == [3, 7, 11]

    @pytest.mark.asyncio
    async def test_async_add_with_invalid_input(self):
        """Test async function with invalid input."""
        with pytest.raises(InvalidInputError):
            await async_add("not", "numbers")


# ============================================================================
# MARKER TESTS
# ============================================================================


class TestMarkers:
    """Tests demonstrating pytest markers."""

    @pytest.mark.skip(reason="Demonstrating skip marker")
    def test_skipped_test(self):
        """This test is always skipped."""
        assert False  # Would fail if run

    @pytest.mark.skipif(
        not hasattr(math, "prod"), reason="math.prod not available in Python < 3.8"
    )
    def test_conditional_skip(self):
        """Skip if math.prod not available."""
        assert math.prod([1, 2, 3, 4]) == 24

    @pytest.mark.xfail(reason="Expected to fail - demonstrating xfail")
    def test_expected_failure(self):
        """This test is expected to fail."""
        assert add(1, 1) == 3  # Wrong expectation

    @pytest.mark.xfail(raises=DivisionByZeroError)
    def test_expected_exception(self):
        """Test expected to raise specific exception."""
        divide(1, 0)

    @pytest.mark.slow
    def test_slow_operation(self):
        """
        Slow test that's skipped by default.
        Run with: pytest --run-slow
        """
        time.sleep(0.1)
        result = factorial(50)
        assert result > 0

    @pytest.mark.smoke
    def test_smoke_basic_add(self):
        """Quick smoke test for basic functionality."""
        assert add(1, 1) == 2

    @pytest.mark.smoke
    def test_smoke_basic_calculator(self):
        """Quick smoke test for Calculator."""
        calc = Calculator(10)
        assert calc.value == 10

    @pytest.mark.regression
    def test_regression_issue_001(self):
        """Regression test for hypothetical bug fix."""
        # Test specific edge case that was previously broken
        result = divide(-10, -2)
        assert result == 5

    @pytest.mark.edge_case
    def test_edge_case_empty_fibonacci(self):
        """Edge case: fibonacci(0)."""
        assert fibonacci(0) == []

    @pytest.mark.edge_case
    def test_edge_case_factorial_zero(self):
        """Edge case: factorial(0) = 1."""
        assert factorial(0) == 1

    @pytest.mark.performance
    def test_performance_fibonacci(self, performance_threshold):
        """Test Fibonacci performance."""
        start = time.perf_counter()
        fibonacci(100)
        duration = time.perf_counter() - start

        assert duration < performance_threshold


# ============================================================================
# FIXTURE DEMONSTRATION TESTS
# ============================================================================


class TestFixturePatterns:
    """Tests demonstrating various fixture patterns."""

    @pytest.mark.unit
    def test_basic_fixture(self, calculator):
        """Using basic function-scoped fixture."""
        assert calculator.value == 0

    @pytest.mark.unit
    def test_fixture_with_value(self, calculator_with_value):
        """Using fixture with initial value."""
        assert calculator_with_value.value == 10

    @pytest.mark.unit
    def test_populated_fixture(self, populated_calculator):
        """Using dependent fixture."""
        # Value should be: (0 + 10) * 2 - 5 = 15
        assert populated_calculator.value == 15

    @pytest.mark.unit
    def test_factory_fixture(self, calculator_factory):
        """Using factory fixture."""
        calc1 = calculator_factory(100)
        calc2 = calculator_factory(200)

        assert calc1.value == 100
        assert calc2.value == 200

    @pytest.mark.unit
    def test_parametrized_fixture(self, various_numbers):
        """Test runs multiple times with different values."""
        result = add(various_numbers, 0)
        assert result == various_numbers

    @pytest.mark.unit
    def test_temp_file_fixture(self, temp_data_file):
        """Using temporary file fixture."""
        import json

        content = temp_data_file.read_text()
        data = json.loads(content)

        assert "values" in data
        assert data["name"] == "test"

    @pytest.mark.unit
    def test_session_fixture(self, large_dataset):
        """Using session-scoped fixture."""
        assert len(large_dataset) == 10000
        assert large_dataset[0] == 1
        assert large_dataset[-1] == 10000

    @pytest.mark.unit
    def test_environment_fixture(self, test_environment):
        """Using environment fixture from command line option."""
        assert test_environment in ["development", "staging", "production"]


# ============================================================================
# DECORATOR TESTS
# ============================================================================


class TestDecorators:
    """Tests for custom decorators."""

    @pytest.mark.unit
    def test_validate_numeric_args_decorator(self):
        """Test the validation decorator."""

        @validate_numeric_args
        def test_func(a, b):
            return a + b

        assert test_func(1, 2) == 3

        with pytest.raises(InvalidInputError):
            test_func("a", "b")

    @pytest.mark.unit
    def test_timing_decorator(self):
        """Test the timing decorator."""

        @timing_decorator
        def slow_func():
            time.sleep(0.01)
            return 42

        result = slow_func()
        assert result == 42

    @pytest.mark.unit
    def test_retry_decorator_success(self):
        """Test retry decorator with eventual success."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.unit
    def test_retry_decorator_failure(self):
        """Test retry decorator with persistent failure."""

        @retry(max_attempts=2, delay=0.01)
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()


# ============================================================================
# PROPERTY-BASED TESTING HINTS
# ============================================================================


class TestPropertyBased:
    """
    Tests demonstrating property-based testing concepts.

    For full property-based testing, consider using hypothesis library.
    These tests show the concept with parametrization.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "a,b", [(i, j) for i in range(-10, 11, 5) for j in range(-10, 11, 5)]
    )
    def test_addition_commutative_property(self, a, b):
        """Property: a + b = b + a"""
        assert add(a, b) == add(b, a)

    @pytest.mark.unit
    @pytest.mark.parametrize("a,b,c", [(1, 2, 3), (0, 0, 0), (-1, 0, 1), (10, 20, 30)])
    def test_addition_associative_property(self, a, b, c):
        """Property: (a + b) + c = a + (b + c)"""
        assert add(add(a, b), c) == add(a, add(b, c))

    @pytest.mark.unit
    @pytest.mark.parametrize("a", range(-10, 11))
    def test_addition_identity_property(self, a):
        """Property: a + 0 = a"""
        assert add(a, 0) == a

    @pytest.mark.unit
    @pytest.mark.parametrize("a,b,c", [(2, 3, 4), (1, 1, 1), (0, 5, 10)])
    def test_multiplication_distributive_property(self, a, b, c):
        """Property: a * (b + c) = (a * b) + (a * c)"""
        left = multiply(a, add(b, c))
        right = add(multiply(a, b), multiply(a, c))
        assert left == right


# ============================================================================
# TEST ORGANIZATION - GROUP BY FUNCTIONALITY
# ============================================================================


class TestEdgeCases:
    """Collection of edge case tests."""

    @pytest.mark.edge_case
    def test_add_with_zero(self):
        """Edge case: adding zero."""
        assert add(0, 0) == 0
        assert add(100, 0) == 100
        assert add(0, 100) == 100

    @pytest.mark.edge_case
    def test_multiply_by_zero(self):
        """Edge case: multiplication by zero."""
        assert multiply(0, 100) == 0
        assert multiply(100, 0) == 0
        assert multiply(0, 0) == 0

    @pytest.mark.edge_case
    def test_divide_zero_by_number(self):
        """Edge case: dividing zero by a number."""
        assert divide(0, 100) == 0
        assert divide(0, -5) == 0

    @pytest.mark.edge_case
    def test_power_edge_cases(self):
        """Edge cases for power function."""
        assert power(0, 0) == 1  # By convention
        assert power(0, 5) == 0
        assert power(5, 0) == 1
        assert power(1, 1000) == 1

    @pytest.mark.edge_case
    def test_statistics_single_value(self):
        """Edge case: statistics with single value."""
        stats = Statistics([42])
        assert stats.mean == 42
        assert stats.median == 42
        assert stats.min == 42
        assert stats.max == 42
        assert stats.range == 0
        assert stats.variance == 0


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.integration
    def test_calculator_with_statistics(self):
        """Test Calculator and Statistics working together."""
        calc = Calculator(10)
        calc.add(20).add(30).add(40).add(50)

        # Use calculator history to build statistics
        # (simplified - just using final value)
        stats = Statistics([10, 30, 60, 100, 150])

        assert stats.count == 5
        assert stats.mean == 70

    @pytest.mark.integration
    def test_math_context_with_calculator(self, math_context):
        """Test MathContext with Calculator operations."""
        calc = Calculator(100)

        with math_context:
            math_context.execute(calc.add, 50)
            assert calc.value == 150

    @pytest.mark.integration
    def test_full_workflow(self):
        """Test a complete mathematical workflow."""
        # Generate data
        data = [factorial(i) for i in range(1, 8)]

        # Analyze with statistics
        stats = Statistics(data)

        # Use calculator for additional computation
        calc = Calculator(stats.mean)
        calc.multiply(2).subtract(stats.min)

        # Verify results
        assert stats.count == 7
        assert calc.value > 0


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run a subset of tests when executed directly
    pytest.main([__file__, "-v", "-m", "unit", "--tb=short"])

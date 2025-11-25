"""
Pytest configuration and fixtures.

This file demonstrates:
- Fixtures with different scopes (function, class, module, session)
- Fixture factories
- Parametrized fixtures
- Autouse fixtures
- Fixture finalization/teardown
- Custom markers
- Pytest hooks
- Command line options
"""

import json
import time
from pathlib import Path
from typing import Callable, Generator, List

import pytest

from my_math import Calculator, MathContext, Statistics

# ============================================================================
# CUSTOM MARKERS REGISTRATION
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests (quick sanity checks)"
    )
    config.addinivalue_line("markers", "regression: marks tests for regression testing")
    config.addinivalue_line("markers", "edge_case: marks tests for edge cases")
    config.addinivalue_line("markers", "performance: marks performance-related tests")


# ============================================================================
# COMMAND LINE OPTIONS
# ============================================================================


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Run slow tests"
    )
    parser.addoption(
        "--performance-threshold",
        action="store",
        default="1.0",
        help="Performance threshold in seconds",
    )
    parser.addoption(
        "--test-env",
        action="store",
        default="development",
        choices=["development", "staging", "production"],
        help="Test environment",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and options."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ============================================================================
# SESSION-SCOPED FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def session_start_time() -> float:
    """Record the start time of the test session."""
    return time.time()


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Create a temporary directory for test data that persists across tests."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def large_dataset() -> List[int]:
    """A large dataset for performance testing. Created once per session."""
    return list(range(1, 10001))


@pytest.fixture(scope="session")
def prime_numbers() -> List[int]:
    """List of prime numbers for testing."""
    primes = []
    for n in range(2, 1000):
        is_prime = True
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return primes


# ============================================================================
# MODULE-SCOPED FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def module_calculator() -> Calculator:
    """
    Calculator instance shared within a module.

    This demonstrates module scope - the same instance
    is reused for all tests in a module.
    """
    calc = Calculator(0)
    yield calc
    # Cleanup after all tests in module complete
    calc.clear()


@pytest.fixture(scope="module")
def sample_statistics_data() -> List[float]:
    """Sample data for statistics tests."""
    return [2.5, 3.5, 4.0, 5.0, 5.5, 6.0, 7.5, 8.0, 9.0, 10.5]


# ============================================================================
# CLASS-SCOPED FIXTURES
# ============================================================================


@pytest.fixture(scope="class")
def class_calculator(request) -> Generator[Calculator, None, None]:
    """
    Calculator instance shared within a test class.

    Demonstrates:
    - Class scope
    - Adding fixture to request.cls for access in class
    - Generator-based fixture with teardown
    """
    calc = Calculator(100)

    # Make available on the test class
    if request.cls is not None:
        request.cls.calculator = calc

    yield calc

    # Teardown
    calc.clear()
    calc.clear_history()


# ============================================================================
# FUNCTION-SCOPED FIXTURES
# ============================================================================


@pytest.fixture
def calculator() -> Calculator:
    """Fresh calculator instance for each test."""
    return Calculator()


@pytest.fixture
def calculator_with_value() -> Calculator:
    """Calculator initialized with value 10."""
    return Calculator(10)


@pytest.fixture
def statistics() -> Statistics:
    """Fresh statistics instance for each test."""
    return Statistics()


@pytest.fixture
def statistics_with_data() -> Statistics:
    """Statistics instance with sample data."""
    return Statistics([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def math_context() -> MathContext:
    """Math context for testing."""
    return MathContext(precision=5, raise_on_error=True)


@pytest.fixture
def silent_math_context() -> MathContext:
    """Math context that doesn't raise errors."""
    return MathContext(precision=5, raise_on_error=False)


# ============================================================================
# PARAMETRIZED FIXTURES
# ============================================================================


@pytest.fixture(params=[0, 1, 10, 100, -5, 3.14])
def various_numbers(request) -> float:
    """Fixture providing various numbers for testing."""
    return request.param


@pytest.fixture(
    params=[
        (1, 2, 3),
        (0, 0, 0),
        (-1, 1, 0),
        (10, 20, 30),
        (0.5, 0.5, 1.0),
    ]
)
def add_test_cases(request) -> tuple:
    """Fixture providing (a, b, expected) tuples for addition tests."""
    return request.param


@pytest.fixture(
    params=[
        ([1, 2, 3], 2.0),
        ([10, 20, 30], 20.0),
        ([5, 5, 5, 5], 5.0),
        ([-10, 0, 10], 0.0),
    ]
)
def mean_test_cases(request) -> tuple:
    """Fixture providing (data, expected_mean) tuples."""
    return request.param


@pytest.fixture(
    params=[
        Calculator,
        Statistics,
    ]
)
def math_class(request):
    """Fixture providing different math classes to test."""
    return request.param


# ============================================================================
# FIXTURE FACTORIES
# ============================================================================


@pytest.fixture
def calculator_factory() -> Callable[[float], Calculator]:
    """
    Factory fixture for creating calculators with specific values.

    Usage in tests:
        def test_something(calculator_factory):
            calc1 = calculator_factory(10)
            calc2 = calculator_factory(20)
    """
    created_calculators: List[Calculator] = []

    def _create_calculator(initial_value: float = 0) -> Calculator:
        calc = Calculator(initial_value)
        created_calculators.append(calc)
        return calc

    yield _create_calculator

    # Cleanup all created calculators
    for calc in created_calculators:
        calc.clear()


@pytest.fixture
def statistics_factory() -> Callable[[List[float]], Statistics]:
    """Factory fixture for creating Statistics with specific data."""

    def _create_statistics(data: List[float] = None) -> Statistics:
        return Statistics(data or [])

    return _create_statistics


# ============================================================================
# AUTOUSE FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def test_timing(request):
    """
    Automatically measure and report test duration.

    autouse=True means this runs for every test without explicit request.
    """
    start_time = time.perf_counter()
    yield
    duration = time.perf_counter() - start_time

    # Add duration as a property (accessible in hooks)
    request.node.test_duration = duration


@pytest.fixture(autouse=True, scope="function")
def reset_logging():
    """Reset logging level before each test."""
    import logging

    logging.getLogger("my_math").setLevel(logging.WARNING)
    yield
    logging.getLogger("my_math").setLevel(logging.WARNING)


# ============================================================================
# FIXTURES WITH TEARDOWN (YIELD vs REQUEST.ADDFINALIZER)
# ============================================================================


@pytest.fixture
def temp_data_file(tmp_path) -> Generator[Path, None, None]:
    """
    Create a temporary JSON file for testing.

    Demonstrates yield-based teardown.
    """
    file_path = tmp_path / "test_data.json"
    data = {"values": [1, 2, 3, 4, 5], "name": "test"}
    file_path.write_text(json.dumps(data))

    yield file_path

    # Teardown: clean up the file
    if file_path.exists():
        file_path.unlink()


@pytest.fixture
def calculator_with_finalizer(request) -> Calculator:
    """
    Calculator fixture using request.addfinalizer.

    Alternative to yield-based teardown.
    Useful when you might have multiple finalizers.
    """
    calc = Calculator(50)

    def cleanup():
        calc.clear()
        calc.memory_clear()

    request.addfinalizer(cleanup)

    return calc


# ============================================================================
# DEPENDENT FIXTURES
# ============================================================================


@pytest.fixture
def populated_calculator(calculator: Calculator) -> Calculator:
    """
    Calculator with operations already performed.

    Depends on the calculator fixture.
    """
    calculator.add(10).multiply(2).subtract(5)
    return calculator


@pytest.fixture
def normalized_statistics(statistics_with_data: Statistics) -> Statistics:
    """
    Statistics fixture with normalized data.

    Depends on statistics_with_data fixture.
    """
    # Normalize data to mean=0, std=1
    mean = statistics_with_data.mean
    std = statistics_with_data.std_dev
    normalized_data = [(x - mean) / std for x in statistics_with_data.data]
    return Statistics(normalized_data)


# ============================================================================
# ASYNC FIXTURES
# ============================================================================


@pytest.fixture
async def async_calculator():
    """Async fixture example."""
    import asyncio

    await asyncio.sleep(0.01)  # Simulate async setup
    calc = Calculator(0)
    yield calc
    await asyncio.sleep(0.01)  # Simulate async teardown


# ============================================================================
# PYTEST HOOKS
# ============================================================================


def pytest_runtest_setup(item):
    """Called before each test setup."""
    # Check if test has environment-specific markers
    envmarker = item.get_closest_marker("env")
    if envmarker:
        env_name = envmarker.args[0]
        current_env = item.config.getoption("--test-env")
        if env_name != current_env:
            pytest.skip(f"Test requires {env_name} environment")


def pytest_runtest_teardown(item, nextitem):
    """Called after each test teardown."""
    pass  # Can be used for cleanup or logging


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to capture test results.

    Demonstrates hookwrapper for accessing both before and after.
    """
    outcome = yield
    report = outcome.get_result()

    # Store test result on the item for later access
    if report.when == "call":
        item.test_result = report.outcome


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary to terminal output."""
    terminalreporter.write_sep("-", "Custom Test Summary")
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    terminalreporter.write_line(f"Tests passed: {passed}")
    terminalreporter.write_line(f"Tests failed: {failed}")


# ============================================================================
# ENVIRONMENT FIXTURES
# ============================================================================


@pytest.fixture
def test_environment(request) -> str:
    """Get the current test environment from command line."""
    return request.config.getoption("--test-env")


@pytest.fixture
def performance_threshold(request) -> float:
    """Get the performance threshold from command line."""
    return float(request.config.getoption("--performance-threshold"))


# ============================================================================
# MOCK/STUB FIXTURES
# ============================================================================


@pytest.fixture
def mock_slow_operation(mocker):
    """
    Mock fixture for slow operations.

    Requires pytest-mock plugin.
    """

    def _mock(target, return_value):
        return mocker.patch(target, return_value=return_value)

    return _mock


@pytest.fixture
def mock_time(mocker):
    """Mock time.time to control timing in tests."""
    mock = mocker.patch("time.time")
    mock.return_value = 1000.0
    return mock

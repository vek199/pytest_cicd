# Pytest & GitHub Actions CI/CD Learning Project

A comprehensive learning project demonstrating pytest testing patterns and GitHub Actions CI/CD workflows.

## Project Structure

```
pytest_cicd/
├── my_math.py           # Main module with math functions and classes
├── test_my_math.py      # Comprehensive test suite
├── conftest.py          # Pytest fixtures, hooks, and configuration
├── pytest.ini           # Pytest configuration
├── setup.cfg            # Tool configuration (flake8, mypy, coverage)
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .github/
    └── workflows/
        └── ci.yml       # GitHub Actions CI/CD pipeline
```

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd pytest_cicd

# Create and activate virtual environment
conda create -n pytest_cicd python=3.12
conda activate pytest_cicd

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

---

## Pytest Features Demonstrated

### 1. Basic Test Functions

```python
def test_add_positive_numbers():
    assert add(1, 2) == 3
```

### 2. Test Classes

```python
class TestCalculator:
    def test_calculator_initialization(self):
        calc = Calculator()
        assert calc.value == 0
```

### 3. Parametrized Tests

Test multiple inputs with a single test function:

```python
@pytest.mark.parametrize("n,expected", [
    (0, 1),
    (1, 1),
    (5, 120),
])
def test_factorial(n, expected):
    assert factorial(n) == expected
```

### 4. Fixtures

#### Function-scoped (default)
```python
@pytest.fixture
def calculator():
    return Calculator()

def test_with_fixture(calculator):
    calculator.add(5)
    assert calculator.value == 5
```

#### Session-scoped (shared across all tests)
```python
@pytest.fixture(scope="session")
def large_dataset():
    return list(range(1, 10001))
```

#### Factory Fixtures
```python
@pytest.fixture
def calculator_factory():
    def _create(value=0):
        return Calculator(value)
    return _create

def test_with_factory(calculator_factory):
    calc = calculator_factory(100)
    assert calc.value == 100
```

### 5. Markers

```python
@pytest.mark.unit          # Unit test
@pytest.mark.slow          # Slow test (skipped by default)
@pytest.mark.skip          # Always skip
@pytest.mark.xfail         # Expected to fail
@pytest.mark.parametrize   # Multiple test cases
```

Run specific markers:
```bash
pytest -m "unit"           # Only unit tests
pytest -m "not slow"       # Exclude slow tests
pytest -m "smoke"          # Only smoke tests
```

### 6. Exception Testing

```python
def test_division_by_zero():
    with pytest.raises(DivisionByZeroError) as excinfo:
        divide(10, 0)
    assert "zero" in str(excinfo.value)
```

### 7. Approximate Comparisons

```python
# For floating point comparisons
assert add(0.1, 0.2) == pytest.approx(0.3)
```

### 8. Mocking

```python
from unittest.mock import Mock, patch

def test_with_mock():
    mock_calc = Mock(spec=Calculator)
    mock_calc.value = 42
    assert mock_calc.value == 42

def test_with_patch():
    with patch('my_math.add') as mock_add:
        mock_add.return_value = 999
        result = mock_add(1, 2)
        assert result == 999
```

### 9. Async Tests

```python
@pytest.mark.asyncio
async def test_async_add():
    result = await async_add(5, 3)
    assert result == 8
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Very verbose (show print statements)
pytest -vv -s

# Run specific file
pytest test_my_math.py

# Run specific test class
pytest test_my_math.py::TestCalculator

# Run specific test function
pytest test_my_math.py::TestCalculator::test_calculator_initialization
```

### Filtering Tests

```bash
# By marker
pytest -m "unit"
pytest -m "not slow"
pytest -m "unit and not integration"

# By keyword in test name
pytest -k "add"
pytest -k "calculator and not memory"

# Failed tests from last run
pytest --lf

# Failed tests first, then others
pytest --ff
```

### Coverage

```bash
# Basic coverage
pytest --cov=.

# With HTML report
pytest --cov=. --cov-report=html

# With terminal missing lines
pytest --cov=. --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=. --cov-fail-under=80
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto        # Auto-detect CPU count
pytest -n 4           # Use 4 workers
```

### Other Options

```bash
# Stop on first failure
pytest -x

# Stop after N failures
pytest --maxfail=3

# Show slowest tests
pytest --durations=10

# Generate HTML report
pytest --html=report.html

# Run with random order
pytest -p randomly

# Run slow tests
pytest --run-slow
```

---

## GitHub Actions CI/CD Pipeline

### Pipeline Jobs

| Job | Description |
|-----|-------------|
| `lint` | Code quality (black, isort, flake8, mypy) |
| `test` | Matrix testing across Python versions and OS |
| `slow-tests` | Long-running tests (scheduled/manual) |
| `performance` | Benchmark tests |
| `docs` | Documentation build |
| `security` | Security scanning (bandit, safety) |
| `ci-success` | Final status check |

### Matrix Strategy

Tests run on:
- **OS**: Ubuntu, macOS, Windows
- **Python**: 3.10, 3.11, 3.12

### Triggers

- **Push**: main, develop, feature/*, release/*
- **Pull Request**: main, develop
- **Schedule**: Daily at midnight UTC
- **Manual**: workflow_dispatch with options

### Manual Workflow Trigger Options

```yaml
inputs:
  run_slow_tests: boolean  # Run slow tests
  debug_enabled: boolean   # Enable debug logging
  python_version: choice   # 3.10, 3.11, 3.12
```

### Artifacts

- Test results (JUnit XML)
- Coverage reports (HTML)
- Benchmark results (JSON)

---

## Code Quality Tools

### Black (Code Formatter)

```bash
# Check formatting
black --check .

# Format code
black .
```

### isort (Import Sorter)

```bash
# Check imports
isort --check-only .

# Sort imports
isort .
```

### Flake8 (Linter)

```bash
flake8 .
```

### Mypy (Type Checker)

```bash
mypy my_math.py
```

---

## Module Features

### Custom Exceptions

```python
MathError          # Base exception
DivisionByZeroError
NegativeNumberError
InvalidInputError
EmptySequenceError
```

### Decorators

```python
@validate_numeric_args  # Validates function arguments
@timing_decorator       # Measures execution time
@retry(max_attempts=3)  # Retries on failure
```

### Calculator Class

```python
calc = Calculator(10)
calc.add(5).multiply(2).subtract(3)
print(calc.value)  # 27

# Memory operations
calc.memory_store()
calc.clear()
calc.memory_recall()

# Method chaining
result = Calculator(0).add(10).multiply(2).value  # 20
```

### Statistics Class

```python
stats = Statistics([1, 2, 3, 4, 5])
print(stats.mean)      # 3.0
print(stats.median)    # 3
print(stats.std_dev)   # 1.414...
print(stats.variance)  # 2.0
print(stats.quartiles())
print(stats.outliers())
```

### Context Manager

```python
with MathContext(precision=2) as ctx:
    result = ctx.execute(divide, 1, 3)
    print(result)  # 0.33
```

---

## Test Organization Best Practices

### By Test Type

```
test_my_math.py
├── TestBasicArithmetic (unit)
├── TestFactorial (unit)
├── TestFibonacci (unit)
├── TestCalculator (unit)
├── TestStatistics (unit)
├── TestExceptions (unit)
├── TestMocking (unit)
├── TestAsyncFunctions (unit)
├── TestMarkers (various)
├── TestEdgeCases (edge_case)
└── TestIntegration (integration)
```

### Fixture Organization (conftest.py)

```
conftest.py
├── Custom Markers
├── Command Line Options
├── Session Fixtures
├── Module Fixtures
├── Class Fixtures
├── Function Fixtures
├── Parametrized Fixtures
├── Factory Fixtures
├── Autouse Fixtures
└── Pytest Hooks
```

---

## Learning Resources

### Pytest

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Markers](https://docs.pytest.org/en/stable/mark.html)

### GitHub Actions

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Actions Marketplace](https://github.com/marketplace?type=actions)

### Python Testing

- [Python Testing with pytest (book)](https://pragprog.com/titles/bopytest2/python-testing-with-pytest-second-edition/)
- [Real Python: Testing](https://realpython.com/pytest-python-testing/)

---

## License

MIT

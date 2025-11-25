from my_math import add, subtract, multiply, divide

def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(-1, -1) == -2

def test_subtract():
    assert subtract(1, 2) == -1
    assert subtract(-1, 1) == -2
    assert subtract(-1, -1) == 0

def test_multiply():
    assert multiply(1, 2) == 2
    assert multiply(-1, 1) == -1
    assert multiply(-1, -1) == 1

def test_divide():
    assert divide(1, 2) == 0.5
    assert divide(-1, 1) == -1
    assert divide(-1, -1) == 1

if __name__ == "__main__":
    test_add()
    test_subtract()
    test_multiply()
    test_divide()
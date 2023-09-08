
fibonacci = lambda n: fibonacci(n - 1) + fibonacci(n - 2) if n > 2 else 1
print(fibonacci(int(input('Enter number for fibonacci calculation: '))))


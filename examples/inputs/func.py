def calculate_average(numbers):
    total = 0
    count = 0
    for n in numbers:
        total = total + n
        count = count + 1
    if count == 0:
        return 0
    result = total / count
    return result

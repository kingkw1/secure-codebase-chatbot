from data_processor.calculate import add, subtract
from data_processor.statistics import mean, median

def main():
    numbers = [1, 2, 3, 4, 5]
    print("Sum:", add(3, 5))
    print("Difference:", subtract(10, 3))
    print("Mean:", mean(numbers))
    print("Median:", median(numbers))

if __name__ == "__main__":
    main()

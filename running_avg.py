from collections import deque
from statistics import mean


class RunningAverage(deque):
    def __init__(self, size=5):
        super().__init__([], size)

    def get_avg(self):
        return mean(self)

    def __lt__(self, value):
        print(avg.get_avg())
        return self.get_avg() < value

    def __gt__(self, value):
        print(avg.get_avg())
        return self.get_avg() > value






if __name__ == "__main__":
    test_data = [5, 6, 30, 3, -10, 9, 8, 8, 4, 3, 1, 0]

    avg = RunningAverage()

    i = 0
    sum_ = 0
    for n in test_data:
        avg.append(n)
        sum_ += n
        i += 1
        k = i
        if i > 5:
            sum_ -= test_data[i-6]
            k = 5
        print(i, k, test_data[i-6], sum_, sum_/k, avg.get_avg())
        assert sum_/k == avg.get_avg(), "sum_/k == {}".format(avg.get_avg())
    print(avg.get_avg(), avg.get_avg())
    assert avg > 3.1, "{} > 3.1".format(avg.get_avg())
    assert 3.1 < avg, "3.1 < {}".format(avg.get_avg())
    assert 3.3 > avg, "3.3 > {}".format(avg.get_avg())
    assert avg < 3.3, "{} < 3.3".format(avg.get_avg())




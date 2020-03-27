def find(intervals):
    res = []
    max_overlap = 0
    intervals.sort(key=lambda x: x[0])
    p = intervals[0]
    for i in range(1, len(intervals)):
        overlap = min(p[1], intervals[i][1]) - intervals[i][0]
        if overlap > max_overlap:
            res = [p, intervals[i]]
            max_overlap = overlap
        if intervals[i][1] > p[1]:
            p = intervals[i]

    return max_overlap, res


if __name__ == '__main__':
    # intervals = [[1, 10], [2, 6], [3,15], [5, 9]]
    # intervals = [(1,6), (3,6), (5,8)]
    intervals = [(1,6), (2,5), (5,8)]
    max_overlap, res = find(intervals)
    print(max_overlap)
    print(res)
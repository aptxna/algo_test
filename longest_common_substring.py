# Space optimized Python3 implementation of Longest Common Substring.

# Function to find longest common substring.
def LCSubStr(X, Y):
    # Find length of both the strings.
    m = len(X)
    n = len(Y)

    # Variable to store length of
    # longest common substring.
    result = 0

    # Matrix to store result of two
    # consecutive rows at a time.
    len_mat = [[0 for i in range(n)] for j in range(2)]

    # Variable to represent which row
    # of matrix is current row.
    currRow = 0

    # For a particular value of i and j,
    # len_mat[currRow][j] stores length of
    # longest common substring in string
    # X[0..i] and Y[0..j].
    for i in range(m):
        for j in range(n):

            if (i == 0 | j == 0):
                len_mat[currRow][j] = 0

            elif (X[i - 1] == Y[j - 1]):

                len_mat[currRow][j] = len_mat[1 - currRow][j - 1] + 1
                result = max(result, len_mat[currRow][j])

            else:
                len_mat[currRow][j] = 0

        # Make current row as previous row and
        # previous row as new current row.
        currRow = 1 - currRow

    return result


# Driver Code
if __name__ == "__main__":
    X = "GeeksforGeeks"
    Y = "GeeksQuiz"

    print(LCSubStr(X, Y))
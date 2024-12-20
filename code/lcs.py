import numpy as np

def lcs(x, y):
    '''
    https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
    '''
    m = len(x)
    n = len(y)
    c = np.zeros((m+1,n+1), dtype=np.int)
    
    # constructing the table (index = 0 will be 0 for all). 
    for i in range(1,m+1): # , leave=True, desc='Aligning'):
        for j in range(1,n+1):
            # if they match, add another in the chain
            if x[i-1] == y[j-1]:
                c[i,j] = c[i-1,j-1] + 1
            # if they don't match, set value to longest chain that's adjacent. 
            # signals that in the past there was a chain of that length that you can get at through this point by moving
            # up or left 
            else:
                c[i,j] = max(c[i,j-1], c[i-1,j])
            # since we set non-matches equal to the adjacents, the only time you get a change when moving from one to the other is when
            # the previous one was a match. so when the values decrease, you can get the value at that entry in the table. 
    
    # getting masks 
    mask1, mask2 = [], []
    i = m 
    j = n
    # backtracking
    while i > 0 and j > 0: 
      # if they match, add indices to your masks
      if x[i-1] == y[j-1]: 
        i-=1
        j-=1
        mask1.append(i)
        mask2.append(j)
      # if moving up one gives a larger value than moving right one, then move up in table (move back in one sequence)
      elif c[i-1][j] > c[i][j-1]: 
        i-=1
      # otherwise move back in other sequence. if equal, doesn't matter which you choose 
      # (will get same length, but maybe diff seq in the end) (see table on wiki)
      else: 
        j-=1
    # mask only contains where they matched
    # this indexing reverses the order
    return mask1[::-1], mask2[::-1]

import sys
import argparse
import numpy as np

def quickgen(n, d, ln_emb, text_file=None):
    a = np.random.randint(0, 10, (1+d+len(ln_emb), n))

    a[0,:] = np.random.randint(0, 2, n)
    for k, x in enumerate(ln_emb):
        a[1+d+k,:]=np.random.randint(0, x, n)
    a = np.transpose(a)
    # print(a)

    ls_str=[]
    for _ in range(d+1):
        ls_str.append("%d")
    for _ in range(len(ln_emb)):
        ls_str.append("%x")
    if text_file is not None:
        np.savetxt(text_file, a, fmt=ls_str, delimiter='\t',)

    return a

if __name__ == "__main__":
    np.random.seed(123)

    n = 4000
    d = 13
    ln_emb = np.array([ 9980333,36084,17217,7378,20134,3,7112,1442,61, 9758201,1333352,313829,10,2208,11156,122,4,970,14, 9994222, 7267859, 9946608,415421,12420,101, 36])
    days = 24
    for k in range(days):
        text_file = "day_" + str(k)
        print(text_file)

        quickgen(n, d, ln_emb, text_file)

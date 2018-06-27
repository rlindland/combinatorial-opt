import numpy as np

graph=np.load('graph.npy')

tour = [6, 9, 1, 0, 2, 4, 8, 3, 7, 5]
length=0
for i in range(len(tour)-1): length+=graph[tour[i],tour[i+1]]
length+=graph[tour[-1], tour[0]]
print(length)
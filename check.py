import numpy as np
import itertools

graph=np.load('graph.npy')

tours=list(itertools.permutations([i for i in range(graph.shape[0])]))
print(type(tours))
min_length=np.sum(graph)
max_length=0
counter=0
avg_length=0
for j in range(len(tours)):
	length=0
	for i in range(len(tours[j])-1): length+=graph[tours[j][i],tours[j][i+1]]
	length+=graph[tours[j][-1], tours[j][0]]
	if length<min_length: 
		best_tour=tours[j]
		min_length=length
	if length>max_length:
		worst_tour=tours[j]
		max_length=length
	avg_length+=length
	counter+=1
	if counter%100000==0:
		print(counter,'/',len(tours),':')
		print('     Best:', min_length, best_tour)
		print('     Worst:', max_length, worst_tour)
		print('     Avg:', avg_length/counter)
print(counter,'/',len(tours),':')
print('     Best:', min_length, best_tour)
print('     Worst:', max_length, worst_tour)
print('     Avg:', avg_length/counter)

print(best_tour, worst_tour)


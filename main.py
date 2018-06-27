import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Q import DQN

B=64
DIM=64
N=500
E=1000
T=4
n=1
EPSILON=.05
GAMMA=1
learning_rate=


def generate_graph():
	n = 10#np.random.randint(50,100)
	matrix = np.random.rand(n,n).astype('float32')
	other_matrix = np.random.rand(n,n).astype('float32')
	matrix = other_matrix.T@matrix.T@matrix@other_matrix
	np.fill_diagonal(matrix, 0)
	np.save('graph.npy', matrix)
	return matrix


def eps(eps): return (np.random.uniform()<eps)

Q=DQN(DIM,T)
mse=nn.MSELoss(size_average=True)
optimizer=optim.SGD([Q.theta1,Q.theta2,Q.theta3,Q.theta4,Q.theta5,Q.theta6,Q.theta7], lr=learning_rate)

M=[]
for episode in range(E):
	if len(M)>N: M=M[-N:]

	graph=generate_graph()

	S=[]
	R=[]
	S_bar=[i for i in range(graph.shape[0])]

	cumloss=0

	for t in range(T):

		if eps(EPSILON): v=S_bar[np.random.randint(len(S_bar))]
		else:
			with torch.no_grad():
				Q.embed(S,graph)
				max_val=Q.q(S_bar[0])
				v=S_bar[0]
				for i in S_bar:
					curr=Q.q(i)
					if curr>max_val:
						max_val=curr
						v=i
		S.append(v)
		S_bar.remove(v)
		if len(R)==0: R.append(0)
		else: R.append(-1*(graph[S[-1], S[-2]]+graph[S[-1],S[0]])+graph[S[-2],S[0]])

		if t>=n:
			Rsum=0
			for i in range(t-n,t+1): Rsum+=R[i]
			M.append((S[:t-n], S[t-n], Rsum, S[:t]))
			for _ in range(B):
				Stn, vt, Rsum, St = M[np.random.randint(len(M))]
				optimizer.zero_grad()
				pred=Q(Stn,vt,graph)
				with torch.no_grad():
					Q.embed(St,graph)
					max_val=Q.q(S_bar[0])
					for i in S_bar:
						curr=Q.q(i)
						if curr>i: max_val=curr
					y=Rsum+GAMMA*max_val
				loss=mse(pred,y)
				loss.backward()
				optimizer.step()
				cumloss += float(loss.data[0])
	print('Epoch:', episode+1,'/', E, '| Epoch Loss:', cumloss)

with torch.no_grad():
	S=[]
	S_bar=[i for i in range(graph.shape[0])]
	for i in range(graph.shape[0]):
		Q.embed(S,graph)
		max_val = Q.q(S_bar[0])
		v=S_bar[0]
		for node in S_bar:
			curr_val = Q.q(node)
			if curr_val>max_val: 
				v=node
				max_val=curr_val
		S.append(v)
		S_bar.remove(v)
	print(S)






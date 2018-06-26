import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Q import Q

DIM = 69
E = 69
T = 69 
N = 1
BATCH_SIZE = 69
EPSILON = .2
GAMMA=1

torch.set_default_tensor_type(torch.DoubleTensor)

Q = Q(DIM,T) 
mse = nn.MSELoss(size_average=False)
optimizer = optim.Adam(Q.parameters())

def generate_graph():
	n = 5#np.random.randint(0,101)
	matrix = np.random.randn(n,n)
	matrix = matrix.T@matrix
	np.fill_diagonal(matrix, 0)
	return matrix


def eps(eps): return (np.random.uniform()<eps)

for _ in range(E):
	M,S,R=[],[],[]
	graph = generate_graph()
	S_bar = [i for i in range(graph.shape[0])]
	cumloss=0
	for t in range(min(T,graph.shape[0])): #change back
		if eps(EPSILON):
			idx = np.random.randint(len(S_bar))
			v = S_bar[idx]
		else:
			with torch.no_grad():
				Q.embed(S,graph)
				max_val = Q.q(S_bar[0])
				v=S_bar[0]
				for node in S_bar: 
					if Q.q(node)>max_val: v=node
		S.append(v)
		S_bar.remove(v)
		if len(R)==0: R.append(0)
		else: R.append(-1*(graph[S[t],v]+graph[v,S[0]]))
		if t>=N: 
			Rsum=0
			for i in range(t-N,t): Rsum+=R[i]
			M.append((S[:t-N], S[t-N], Rsum, S[:t]))
			B = []
			for _ in range(min(BATCH_SIZE,t-N)): #change back
				Q.zero_grad()
				S_t_n, v_t_n, R_tn, S_t = M[np.random.randint(len(M))]
				pred = Q(S_t_n,v_t_n,graph)
				with torch.no_grad():
					Q.embed(S_t,graph)
					max_val = Q.q(0)
					for node in S_bar:
						Qtn=Q.q(node) 
						if Qtn>max_val: max_val=Qtn
					y=R_tn+max_val*GAMMA
				loss = mse(pred, y)
				loss.backward()
				optimizer.step()
				cumloss += float(loss.data)
	print('Epoch:', t+1,'/', E, '| Epoch Loss:', cumloss)













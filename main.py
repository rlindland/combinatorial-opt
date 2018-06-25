import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from q import Q

DIM = 69
E = 69
T = 69 
N = 69
BATCH_SIZE = 69
EPSILON = .69
GAMMA=.69

Q = Q(DIM,T) 
mse = nn.MSELoss(size_average=False)
optimizer = optim.Adam(model.parameters())

def generate_graph():
	n = np.random.randint(100,1000)
	matrix = np.random.randn(n,n)
	matrix = matrix.T@matrix
	np.fill_diagonal(matrix, 0)

def eps(eps): return (np.random.uniform()<eps)

for _ in range(E):
	M,S,R=[],[],[]
	graph = generate_graph()
	S_bar = [i for i in range(graph.shape[0])]
	v=0
	cumloss=0
	for t in range(T):
		if eps(EPSILON):
			idx = np.random.randint(len(S_bar))
			v = S_bar[idx]
		else:
			with torch.no_grad():
				Q.embed(S,graph)
				max_val = Q.q(S_bar[0])
				for node in S_bar: 
					if Q.q(node)>max_val: v=node
		S.append(v)
		S_bar.remove(v)
		if len(R)=0: R.append(0)
		else: R.append(graph[S[t],v])
		if t>=N: 
			Rsum=0
			for i in range(t-N,t): Rsum+=R[i]
			M.append((S[:t-n], S[t-n], Rsum, S[:t]))
			B = []
			for _ in range(BATCH_SIZE):
				Q.zero_grad()
				S_t_n, v_t_n, R_tn, S_t = M[np.random.randint(len(M))]
				pred = Q(S_t_n,v_t_n,graph)
				with torch.no_grad():
					Q.embed(S_t,graph)
					max_val = Q.q(S_bar[0])
					for node in S_bar:
						Qtn=Q.q(node) 
						if Qtn>max_val: max_val=Qtn
					y=R_tn+max_val*GAMMA
				loss = mse(pred, y)
				loss.backward()
				optimizer.step()
				cumloss += float(loss.data)
	print('Epoch:', T+1,'/', EPOCH, '| Epoch Loss:', cumloss)













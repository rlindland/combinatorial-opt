import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Q(nn.Module):

	def __init__(self, dim, T):
		self.T = T
		self.dim = dim

		self.theta1 = nn.Linear(1,dim,bias=False)
		self.theta2 = nn.Linear(dim,dim,bias=False)
		self.theta3 = nn.Linear(dim,dim,bias=False)
		self.theta4 = nn.Linear(1,dim,bias=False)
		self.theta5 = nn.Linear(2*dim,1,bias=False)
		self.theta6 = nn.Linear(dim,dim,bias=False)
		self.theta7 = nn.Linear(dim,dim,bias=False)

	def embed(self,S,graph):
		graph = torch.from_numpy(graph)
		self.u = torch.zeros(self.dim,graph.shape[0])
		for _ in range(self.T):
			u_prime=[]
			summ = u@(torch.ones(u.shape[0])).view(u.shape[0],1)
			for v in range(graph.shape[0]):
				theta1xv=self.theta1(x[v])
				theta2sum=self.theta2(summ-u[:,v])
				theta3sum=0 #self.theta3(F.relu(self.theta4(graph[v,:]@torch.ones(graph.shape[0])-graph[v,v])))
				for v_prime in range(graph.shape[0]): theta3sum+=F.relu(self.theta4(graph(v,v_prime)))
				theta3sum=self.theta3(theta3sum)
				u_prime.append(F.relu(theta1xv+theta2sum+theta3sum))
			self.u=torch.t(torch.tensor(u_prime))

	def q(self,v):
		x = [1 if v in S else 0 for v in range(graph.shape[0])]
		theta6sum=self.theta6(self.u@torch.ones(self.u.shape[0]))
		theta7uv=self.theta7(self.u[:,v])
		return float(self.theta5(F.relu(torch.cat(theta6sum,theta7uv))))

	def forward(self,S,v,graph):
		self.embed(S,graph)
		return self.q(v)
















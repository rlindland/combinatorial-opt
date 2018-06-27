import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DQN(nn.Module):

	def __init__(self, dim, T):
		super(DQN, self).__init__()
		self.dim=dim
		self.T=T

		self.theta1 = torch.randn(1,dim,requires_grad=True)
		self.theta2 = torch.randn(dim,dim,requires_grad=True)
		self.theta3 = torch.randn(dim,dim,requires_grad=True)
		self.theta4 = torch.randn(1,dim,requires_grad=True)
		self.theta5 = torch.randn(2*dim,1,requires_grad=True)
		self.theta6 = torch.randn(dim,dim,requires_grad=True)
		self.theta7 = torch.randn(dim,dim,requires_grad=True)

	def embed(self,S,graph):
		self.graph = torch.from_numpy(graph)
		self.u = torch.zeros(graph.shape[0], self.dim)

		x = [1.0 if i in S else 0.0 for i in range(graph.shape[0])]

		for _ in range(self.T):
			u_prime = []
			summ=self.u.sum(dim=0,keepdim=True)
			for v in range(graph.shape[0]):
				a=x[v]*self.theta1
				b=(summ-self.u[v])@self.theta2
				c=0
				for u in range(graph.shape[0]): c+=F.relu(self.theta4*float(graph[v,u]))
				c=c@self.theta3
				u_prime.append(F.relu(a+b+c))
			self.u=torch.cat(u_prime)

	def q(self,v):
		summ=self.u.sum(dim=0)
		a=summ@self.theta6
		b=self.u[v]@self.theta7
		c=torch.cat((a,b))
		c=F.relu(c)@self.theta5
		return c

	def forward(self,S,v,graph):
		self.embed(S,graph)
		return self.q(v)
		








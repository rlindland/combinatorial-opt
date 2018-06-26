import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Q(nn.Module):

	def __init__(self, dim, T):
		super(Q, self).__init__()

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
		self.u = torch.zeros(graph.shape[0],self.dim)
		x = torch.tensor([1.0 if v in S else 0.0 for v in range(graph.shape[0])])
		for _ in range(self.T):
			u_prime=[]
			summ = torch.ones(1,self.u.shape[0])@self.u
			for v in range(graph.shape[0]):
				theta1xv=self.theta1(x[v].view(1,1))
				theta2sum=self.theta2(summ-self.u[v,:])
				theta3sum=0 
				for v_prime in range(graph.shape[0]): theta3sum+=F.relu(self.theta4(graph[v,v_prime].view(1,1)))#;print('3:', theta3sum.shape)
				theta3sum=self.theta3(theta3sum)
				u_prime.append(F.relu(theta1xv+theta2sum+theta3sum))
			self.u=torch.cat(u_prime)

	def q(self,v):
		theta6sum=self.theta6(torch.ones(1,self.u.shape[0])@self.u)
		theta7uv=self.theta7(self.u[v,:].view(1,-1))
		return self.theta5(F.relu(torch.cat((theta6sum,theta7uv),dim=1)))

	def forward(self,S,v,graph):
		self.embed(S,graph)
		return self.q(v)
















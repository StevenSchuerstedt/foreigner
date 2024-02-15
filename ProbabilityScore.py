import torch
import torch.nn.functional as F
import numpy as np

class ProbabilityScore(torch.nn.Module):
  def __init__(self):
    super().__init__()
    #self.tau = torch.nn.Parameter(torch.Tensor([1.0]))
    #self.lámbda = torch.nn.Parameter(torch.Tensor([0.0]))

    self.tau = torch.nn.Parameter(torch.Tensor([2.3151]))
    self.lámbda = torch.nn.Parameter(torch.Tensor([-0.0030]))

    #-38.3947
    #8.2753
  def forward(self, similarity_scores):
    P = torch.tensor(np.ones(len(similarity_scores)))
           
    for i in range(len(similarity_scores)):
        sorted = np.flip(np.sort(similarity_scores))

        #TODO: change to softplus / relu
        n = F.relu(torch.exp(torch.tensor(similarity_scores[i] - sorted[0])/self.tau) - self.lámbda)
                       
        A = torch.exp( torch.tensor((sorted - sorted[0]))/self.tau )
        B = F.relu( A- self.lámbda)
        d = torch.sum(B)
                       
        P[i] = n / d
    #sum of P should be 1
    return P
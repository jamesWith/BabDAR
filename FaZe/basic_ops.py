import torch
import math


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    #def __init__(self, consensus_type, dim=1):
    #    self.consensus_type = consensus_type
    #    self.dim = dim
    #    self.shape = None

    @staticmethod
    def forward(ctx, input, consensus_type, dim=1):
        shape = input.size()
        ctx.save_for_backward(input)
        ctx.consensus_type = consensus_type
        ctx.dim = dim
        ctx.shape = shape
        if consensus_type == 'avg':
            output = input.mean(dim=dim, keepdim=True)
        elif consensus_type == 'identity':
            output = input
        else:
            output = None

        return output

    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        consensus_type = ctx.consensus_type
        dim = ctx.dim
        shape = ctx.shape
        if consensus_type == 'avg':
            grad_in = grad_output.expand(shape) / float(shape[dim])
        elif consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in, None, None


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus.apply(input, self.consensus_type, self.dim)
        #return SegmentConsensus(self.consensus_type, self.dim)(input)

'''


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    #def __init__(self, consensus_type, dim=1):
    #    self.consensus_type = consensus_type
    #    self.dim = dim
    #    self.shape = None

    @staticmethod
    def forward(ctx, input_tensor):
        
        shape = input_tensor.size()
        ctx.save_for_backward(input_tensor, shape)
        #if self.consensus_type == 'avg':
        output = input_tensor.mean()
        #elif self.consensus_type == 'identity':
        #    output = input_tensor
        #else:
        #    output = None

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, dim, shape = ctx.saved_tensors
        #if self.consensus_type == 'avg':
        grad_in = grad_output.expand(shape) / float(shape[dim])
            #grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        #elif self.consensus_type == 'identity':
        #    grad_in = grad_output
        #else:
        #    grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    #def __init__(self, consensus_type, dim=1):
    #    super(ConsensusModule, self).__init__()
    #    self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
    #    self.dim = dim
    @staticmethod
    def forward(input):
        #return SegmentConsensus(self.consensus_type, self.dim)(input)
        return SegmentConsensus()(input)
'''

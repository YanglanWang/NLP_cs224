#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch.nn as nn
import torch
class Highway(nn.Module):
    def __init__(self,e_word):
        super(Highway,self).__init__()
        self.e_word=e_word
        self.W_proj=nn.Linear(self.e_word,self.e_word,bias=True)
        self.W_gate=nn.Linear(self.e_word,self.e_word,bias=True)

    def forward(self, X_conv_out):
        X_proj_tmp=self.W_proj(X_conv_out)
        X_proj =X_proj_tmp.clamp(min=0)
        X_gate_tmp=self.W_gate(X_conv_out)
        X_gate=torch.nn.functional.softmax(X_gate_tmp,dim=1)
        X_highway=X_gate*X_proj+(torch.ones(X_gate.size())-X_gate)*X_conv_out
        return X_highway


### END YOUR CODE 


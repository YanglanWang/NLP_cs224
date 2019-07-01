#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch
class CNN(nn.Module):
    def __init__(self,e_char,k,m_word,output_channels):
        super(CNN,self).__init__()
        self.e_char=e_char
        self.k=k
        self.m_word=m_word
        self.in_channels=e_char
        self.output_channels=output_channels


        self.Conv1d=nn.Conv1d(self.in_channels,self.output_channels,self.k,bias=True)
        self.Maxpool1d=nn.MaxPool1d(self.m_word-self.k+1)

    def forward(self, X_reshaped):
        X_conv=self.Conv1d(X_reshaped)
        X_conv_out=self.Maxpool1d(X_conv.clamp(min=0))
        X_conv_out=torch.squeeze(X_conv_out,dim=-1)
        return X_conv_out


### END YOUR CODE


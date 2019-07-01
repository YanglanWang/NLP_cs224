from highway import Highway
import torch

BATCH_SIZE = 5
EMBED_SIZE = 3
def question_1h_sanity_check(model,X_conv_out):
    print('Running Sanity Check for question 1h: highway, coded by myself!!\n')
    print('-'*80)
    X_highway=model.forward(X_conv_out)
    print("predict highway is  "+str(X_highway.size())+'\n'+str(X_highway)+'\n')
    print("the intermediate values and weights are as followed:\n")
    print('X_conv_out'+str(X_conv_out)+'\n')
    X_proj = model.W_proj(X_conv_out).clamp(min=0)
    X_gate_tmp = model.W_gate(X_conv_out)
    X_gate = torch.nn.functional.softmax(X_gate_tmp, dim=1)
    print("W_proj_tmp"+str(model.W_proj.weight)+'  '+"b_proj"+str(model.W_proj.bias)+'\n')
    print("X_proj:"+str(X_proj.size())+"  "+str(X_proj)+'\n')
    print("W_gate"+str(model.W_gate.weight)+'  '+"b_gate"+str(model.W_gate.bias)+'\n')
    print("X_gate_tmp:"+str(X_gate_tmp.size())+"  "+str(X_gate_tmp)+'\n')
    print("X_gate:"+str(X_gate.size())+"  "+str(X_gate)+'\n')
    if X_highway.size()==(BATCH_SIZE,EMBED_SIZE):
        print('test pass')
    else:
        print("test fail")




X_conv_out=torch.rand(BATCH_SIZE,EMBED_SIZE)

if __name__=='__main__':
    model=Highway(EMBED_SIZE)
    X_conv_out=torch.rand(BATCH_SIZE,EMBED_SIZE)
    question_1h_sanity_check(model,X_conv_out)
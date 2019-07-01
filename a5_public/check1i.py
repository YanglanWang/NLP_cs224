from cnn import CNN
import torch.nn as nn
import torch
E_CHAR=50
KERNEL_SIZE=5
M_WORD=21
EMBED_SIZE = 3
NUM_OUTPUT_CHANNELS=EMBED_SIZE
BATCH_SIZE = 5


def question_1i_sanity_check(model,X_reshaped):
    print('Running Sanity Check for question 1h: highway, coded by myself!!\n')
    print('-' * 80)
    X_conv_out=model.forward(X_reshaped)
    print("X_reshaped:\n"+str(X_reshaped)+'\n')
    print("conv1d weight:\n"+str(model.Conv1d.weight)+'\n')
    print("conv1d bias:\n"+str(model.Conv1d.bias)+'\n')
    print("X_conv_out:\n"+str(X_conv_out)+'\n')
    X_conv=model.Conv1d(X_reshaped)
    print("X_conv:\n"+str(X_conv)+'\n')
    if X_conv_out.shape==(BATCH_SIZE,EMBED_SIZE):
        test=[]
        for t in range(M_WORD-k+1):
            a=(X_reshaped[4,:,t:t+k])*model.Conv1d.weight[2,:,:]
            test.append(torch.sum(a)+model.Conv1d.bias[2])
        print(str(test)+"\n")
        print("test pass\n")
    else:
        print("test fail\n")




if __name__=='__main__':
    e_char=E_CHAR
    k=KERNEL_SIZE
    m_word=M_WORD
    output_channels=NUM_OUTPUT_CHANNELS

    X_reshaped=torch.rand(BATCH_SIZE,E_CHAR,M_WORD)
    model=CNN(e_char,k,m_word,output_channels)
    question_1i_sanity_check(model,X_reshaped)

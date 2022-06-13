from embeddingsynth import *
import torch
import sys
code = CodeParrot.small()
#code = CodeGen.small()

#embeds = torch.cat([code.inputs2embeds("def"),torch.randn(code.dim,device=code.model.device)[None,:].exp()])
#inputs = torch.randint(0,code.tokens,(12,))

with code.eval():
    str = "def calculate_epsilon_of_apocalypse():"
    #print(str)
    sys.stdout.write(str)
    for logits, topk in code.generate(code.inputs2embeds(str),output=code.logits2topk):
        #print(topk)
        sys.stdout.write(topk[0])
        sys.stdout.flush()

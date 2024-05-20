#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from networks.UTFormer import UTFormer

if __name__ == "__main__":
    input_shape     = [256, 256, 3]
    num_classes     = 6
    phi             = 'b0'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = UTFormer(num_classes = num_classes, phi = phi, pretrained=False).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    flops           = flops
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

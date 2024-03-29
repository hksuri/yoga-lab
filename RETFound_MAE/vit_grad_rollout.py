'''
Credits to the original author of this code: https://github.com/jacobgil/vit-explain/tree/main
'''

import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def grad_rollout(attentions, gradients, discard_ratio):
    # print(f'\n\nattention length: {len(attentions)}')
    # print(f'gradient length: {len(gradients)}')
    # print(f'attentions size: {attentions[0].size()}')
    # print(f'gradient size: {gradients[0].size()}')
    result = torch.eye(attentions[0].size(-1))
    # print(f'result size: {result.size()}')
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):                
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0
            # print(f'attention size: {attention.size()}')
            # print(f'grad size: {grad.size()}')
            # print(f'attention_heads_fused size: {attention_heads_fused.size()}')

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0
            attention_heads_fused = flat.view(attention_heads_fused.size())

            # I = torch.eye(attention_heads_fused.size(-1))
            # a = (attention_heads_fused + 1.0*I)/2
            # a = a / a.sum(dim=-1, keepdim=True)
            # if a.size(0) != 1:
            #     a = a.mean(dim=0, keepdim=True)
            # print(f'a size: {a.size()}')
            # result = torch.matmul(a, result)
            result = attention_heads_fused
    
    # Look at the total attention between the class token,
    # and the image patches
    # print(f'result size after: {result.size()}')
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        _,output = self.model(input_tensor)
        category_mask = torch.zeros(output.size()).cuda()
        category_mask[:, category_index] = 1
        loss = (output*category_mask).sum()
        loss.backward()

        return grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio)
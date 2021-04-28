import torch
import torch.nn.functional as F


def write_tensor_to_txt(filename, t):
    with open(filename, 'w') as f:
        f.write(str(len(t)))
        f.write('\n')
        for i in range(len(t)):
            f.write(str(t[i].item()))
            f.write('\n')


img_w = 32
img_h = 32
chan = 3

img_size = img_h * img_w * chan

img = torch.rand([img_size])

write_tensor_to_txt('img.txt', img)

ker_w = 4
ker_h = 4
chan_out = 4
stride = 1
dilation = 1
padding = 2

ker = torch.randn([chan * ker_h * ker_w * chan_out])

write_tensor_to_txt('ker.txt', ker)

img_r = img.reshape([1, img_h, img_w, chan]).permute([0, 3, 1, 2])
ker_r = ker.reshape([chan, ker_h, ker_w, chan_out]).permute([3, 0, 1, 2])

out_r = F.conv2d(img_r, ker_r, None, stride, padding, dilation)


out = out_r.squeeze(0).permute([1, 2, 0]).flatten()



write_tensor_to_txt('out.txt', out)

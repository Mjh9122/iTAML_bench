import torch.nn as nn
import torch.nn.functional as F

class VersaConvBlock(nn.Module):
    """
    VERSA-style conv block:
      Conv2d (3x3, same padding) → BatchNorm (optional) → ReLU → Dropout → MaxPool(2x2)
    - Uses Xavier normal init for conv weights to mirror the TF xavier initializer.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 use_batch_norm=True, dropout_p=0.1, pool_kernel=2):
        super().__init__()
        bias = not use_batch_norm  # mirror TF use_bias=False when BN is used
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        if use_batch_norm:
            # Match existing BN behavior in this repo: momentum=1.0, no running stats
            self.bn = nn.BatchNorm2d(out_ch, momentum=1.0, affine=True, track_running_stats=False)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.pool_kernel = pool_kernel

        # Xavier normal to match TF xavier_initializer_conv2d(uniform=False)
        nn.init.xavier_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, self.pool_kernel)  # 'valid' padding to preserve existing output dims
        return x
    
class NonStationaryEncoder(nn.Module):
    """
    VERSA-style 4-layer ConvNet feature extractor for the 'non_stationary' dataset.
    Architecture per block: Conv(3x3, 64ch) → BN → ReLU → Dropout → MaxPool(2x2).
    - Keeps the same I/O characteristics as the previous NonStationaryEncoder:
      for 28x28 inputs and 4 pools → (B, 64) flattened output.
    """
    def __init__(self, num_in_ch=3, num_filter=64, num_conv_layer=4,
                 kernel_size=3, stride=1, padding=1, maxpool_kernel_size=2,
                 input_size=28, use_batch_norm=True, dropout_p=0.1):
        super().__init__()
        ch = [num_in_ch] + [num_filter] * num_conv_layer
        blocks = []
        for i in range(num_conv_layer):
            blocks.append(VersaConvBlock(
                in_ch=ch[i],
                out_ch=ch[i+1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,            # 'same' conv via padding=1 for 3x3
                use_batch_norm=use_batch_norm,
                dropout_p=dropout_p,
                pool_kernel=maxpool_kernel_size,
            ))
        self.encoder = nn.Sequential(*blocks)

        # compute final spatial size after num_conv_layer pools (valid pooling to match prior behavior)
        def down(sz, n, k=2):
            for _ in range(n):
                sz = sz // k
            return max(1, int(sz))

        h_out = down(input_size, num_conv_layer, maxpool_kernel_size)
        w_out = down(input_size, num_conv_layer, maxpool_kernel_size)
        self.out_dim = num_filter * h_out * w_out  # for 28x28 → 64*1*1 = 64

    def forward(self, x):
        h = self.encoder(x)                 # (B, 64, H', W')
        return h.view(h.size(0), -1)        # flatten
    
    
class NonStationaryClassifier(nn.Module):
    def __init__(self, num_in_ch=3, num_filter=64, num_conv_layer=4,
                 kernel_size=3, stride=1, padding=1, maxpool_kernel_size=2,
                 input_size=28, use_batch_norm=True, dropout_p=0.1, classes = 5):
        super().__init__()
        self.conv = NonStationaryEncoder(num_in_ch, num_filter, num_conv_layer,
                 kernel_size, stride, padding, maxpool_kernel_size,
                 input_size, use_batch_norm, dropout_p)
        self.fc = nn.Linear(self.conv.out_dim, classes, bias=False)
    
    def forward(self, x):
        x = self.conv(x)
        
        x1 = self.fc(x)
        x2 = self.fc(x)
        
        return x2, x1
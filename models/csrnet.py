import torch
import torch.nn as nn
import torch.nn.functional as F

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        # VGG-16 frontend configuration (first 10 layers with maxpooling)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # Dilated Convolution Backend (fully convolutional)
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        
        # Frontend does not use dilation (standard convolutions)
        self.frontend = make_layers(self.frontend_feat, in_channels=3, dilation=False) 
        # Backend uses dilation
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True) 
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1) # Output a single channel density map
        
        if not load_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels, batch_norm=False, dilation=False):
    layers = []
    
    # Define dilation rates for the backend.
    # Frontend (dilation=False) will use padding=1 (kernel_size=3, dilation=1)
    # Backend (dilation=True) will use the specified dilation rates
    d_rates = [2, 2, 2, 4, 4, 4] # Common dilation rates for CSRNet backend
    dilation_idx = 0
    
    for v in cfg:
        if v == 'M': # Max Pooling layer
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else: # Convolutional layer
            # Determine padding and dilation based on the 'dilation' flag
            if dilation:
                # Ensure dilation_idx doesn't go out of bounds for d_rates
                if dilation_idx < len(d_rates):
                    current_dilation = d_rates[dilation_idx]
                    dilation_idx += 1 # Increment only for dilated conv layers
                else:
                    # Fallback if d_rates is exhausted (shouldn't happen with correct config)
                    current_dilation = 1
            else:
                current_dilation = 1 # No dilation for frontend layers
            
            # For kernel_size=3, padding = dilation to maintain feature map size
            current_padding = current_dilation 
            
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=current_padding, dilation=current_dilation)
            in_channels = v
            
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            
    return nn.Sequential(*layers)
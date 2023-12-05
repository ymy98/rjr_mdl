import torch 



class SingleCNNModel(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kennel_size):
        super().__init__()
        self.conv_layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_channel, output_channel, kennel_size, stride=1, padding=0))
        # self.relu_layer1 = torch.nn.Sequential(torch.nn.ReLU())
        # self.conv_layer2 = torch.nn.Sequential(torch.nn.Conv2d(16, 256, 3, stride=1, padding=0))

    def forward(self, x):
        x_0 = self.conv_layer1(x)
        # x_1 = self.relu_layer1(x_0)
        # x_2 = self.conv_layer2(x_1)

        return x_0
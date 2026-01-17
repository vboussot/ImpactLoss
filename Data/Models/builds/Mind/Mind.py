
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class Normalize(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if stats.shape[0] == 4:
            return (x - stats[0]) / (stats[1] - stats[0] + 1e-6)
        else:
            vmin = x.min()
            return (x - vmin) / (x.max() - vmin + 1e-6)
        
def pdist_squared3D(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist

class Mind3D(torch.nn.Module):
    def __init__(self, radius=2, dilation=2):
        super().__init__()
        self.radius = radius
        self.dilation = dilation
        six_neighbourhood = torch.Tensor([[0,1,1],
                                [1,1,0],
                                [1,0,1],
                                [1,1,2],
                                [2,1,1],
                                [1,2,1]]).long()
        
        dist = pdist_squared3D(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
        mask = ((x > y).view(-1) & (dist == 2).view(-1))
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
        
        mshift1 = torch.zeros(12, 1, 3, 3, 3)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1

        self.conv1 = nn.Conv3d(1, 12, kernel_size=3, stride=1, padding=1, bias=False, dilation=dilation, groups=1)
        self.conv2 = nn.Conv3d(1, 12, kernel_size=3, stride=1, padding=1, bias=False, dilation=dilation, groups=1)

        self.conv1.weight = nn.Parameter(mshift1, requires_grad=False)
        self.conv2.weight = nn.Parameter(mshift2, requires_grad=False)

        self.rpad1 = nn.ReplicationPad3d(self.dilation)
        self.rpad2 = nn.ReplicationPad3d(self.radius)
        self.normalize = Normalize()
    
    def forward(self, x: torch.Tensor, nb_layers: torch.Tensor = torch.tensor([1]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        x = self.normalize(x, stats)
        ssd = F.avg_pool3d(
            self.rpad2(
                (self.conv1(self.rpad1(x)) - self.conv2(self.rpad1(x))) ** 2
            ),
            self.radius * 2 + 1, stride=1
        )  
        mind = ssd - torch.min(ssd, dim=1, keepdim=True)[0]
        mind_var = torch.mean(mind, dim=1, keepdim=True)
        mind_var_mean = mind_var.mean()
        mind_var = torch.clamp(mind_var, mind_var_mean * 0.001, mind_var_mean * 1000)
        mind /= mind_var
        mind = torch.exp(-mind)
        return [mind]


def pdist_squared2D(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, float('inf'))
    return dist


class Mind2D(nn.Module):
    def __init__(self, radius=2, dilation=2):
        super().__init__()
        self.radius = radius
        self.dilation = dilation

        cross_neigh = torch.tensor([
            [1, 0],
            [0, 1],
            [1, 2],
            [2, 1]
        ]).long()
        dist = pdist_squared2D(cross_neigh.t().unsqueeze(0)).squeeze(0)
        x, y = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        idx_shift1 = cross_neigh.unsqueeze(1).repeat(1, 4, 1).view(-1, 2)[mask, :]
        idx_shift2 = cross_neigh.unsqueeze(0).repeat(4, 1, 1).view(-1, 2)[mask, :]
        num_pairs = idx_shift1.shape[0]

        mshift1 = torch.zeros(num_pairs, 1, 3, 3)
        mshift2 = torch.zeros(num_pairs, 1, 3, 3)
        mshift1.view(-1)[torch.arange(num_pairs) * 9 + idx_shift1[:, 0] * 3 + idx_shift1[:, 1]] = 1
        mshift2.view(-1)[torch.arange(num_pairs) * 9 + idx_shift2[:, 0] * 3 + idx_shift2[:, 1]] = 1

        self.conv1 = nn.Conv2d(1, num_pairs, kernel_size=3, stride=1, padding=1,
                               bias=False, dilation=dilation, groups=1)
        self.conv2 = nn.Conv2d(1, num_pairs, kernel_size=3, stride=1, padding=1,
                               bias=False, dilation=dilation, groups=1)

        self.conv1.weight = nn.Parameter(mshift1, requires_grad=False)
        self.conv2.weight = nn.Parameter(mshift2, requires_grad=False)

        self.rpad1 = nn.ReplicationPad2d(dilation)
        self.rpad2 = nn.ReplicationPad2d(radius)
        self.normalize = Normalize()
        
    def forward(self, x: torch.Tensor, nb_layers: torch.Tensor = torch.tensor([1]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        x = self.normalize(x, stats)
        ssd = F.avg_pool2d(
            self.rpad2((self.conv1(self.rpad1(x)) - self.conv2(self.rpad1(x))) ** 2),
            self.radius * 2 + 1,
            stride=1
        )
        mind = ssd - torch.min(ssd, dim=1, keepdim=True)[0]
        mind_var = torch.mean(mind, dim=1, keepdim=True)
        mind_var_mean = mind_var.mean()
        mind_var = torch.clamp(mind_var, mind_var_mean * 0.001, mind_var_mean * 1000)
        mind /= mind_var
        mind = torch.exp(-mind)
        return [mind]

if __name__ == "__main__":
    for r in [1,2]:
        for d in [1,2]:
            model = Mind2D(r,d)
            
            traced_script_module = torch.jit.script(model)
            traced_script_module.save("./R{}D{}_2D.pt".format(r, d))

            model = Mind3D(r,d)
            traced_script_module = torch.jit.script(model)
            traced_script_module.save("./R{}D{}_3D.pt".format(r, d))

import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import Resnet18

# ConvBNReLU: Konvolyutsiya + Batch Normalization + ReLU
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv.weight, a=1)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

# BiSeNetOutput: BiSeNetning chiqishi
class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        return self.conv_out(self.conv(x))

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv_out.weight, a=1)

# AttentionRefinementModule: E'tiborni optimallashtirish moduli
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.sigmoid_atten(self.bn_atten(self.conv_atten(atten)))
        return torch.mul(feat, atten)

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv_atten.weight, a=1)

# ContextPath: Kontekst yo‘li (ResNet bilan)
class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg_up = F.interpolate(self.conv_avg(avg), size=feat32.size()[2:], mode='nearest')

        feat32 = self.arm32(feat32) + avg_up
        feat16 = self.arm16(feat16) + F.interpolate(feat32, size=feat16.size()[2:], mode='nearest')

        return feat8, feat16, feat32

# FeatureFusionModule: Xususiyatlarni birlashtirish moduli
class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        # Ensure fcp is resized to match the size of fsp
        if fcp.size(2) != fsp.size(2) or fcp.size(3) != fsp.size(3):
            fcp = F.interpolate(fcp, size=fsp.shape[2:], mode='bilinear', align_corners=True)

        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)

        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)

        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

# BiSeNet: Asosiy model
class BiSeNet(nn.Module):
    def __init__(self, n_classes):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

if __name__ == "__main__":
    net = BiSeNet(19)
    net.eval()
    in_ten = torch.randn(16, 3, 640, 480).cuda()
    out, out16, out32 = net(in_ten)

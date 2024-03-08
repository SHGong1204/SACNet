import torch.nn.functional as F
import torch.nn as nn
from mmcv.cnn import kaiming_init
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
from ..builder import NECKS
import torch
import numpy as np
import cv2
import math

class build_new(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feas, atts):
        New_feas = []
        shape_0 = feas[0].shape[2:]
        shape_1 = feas[1].shape[2:]
        shape_2 = feas[2].shape[2:]
        shape_3 = feas[3].shape[2:]

        fea_1_0 = F.interpolate(feas[1], size=shape_0, mode='nearest')
        att_1_0 = F.interpolate(atts[1], size=shape_0, mode='nearest') * atts[0]
        fea_2_0 = F.interpolate(feas[2], size=shape_0, mode='nearest')
        att_2_0 = F.interpolate(atts[2], size=shape_0, mode='nearest') * atts[0]
        fea_3_0 = F.interpolate(feas[3], size=shape_0, mode='nearest')
        att_3_0 = F.interpolate(atts[3], size=shape_0, mode='nearest') * atts[0]

        New_fea_0 = feas[0] * (1 + atts[0]) + (fea_1_0 * att_1_0) + (fea_2_0 * att_2_0) + (fea_3_0 * att_3_0)

        fea_0_1 = F.interpolate(feas[0], size=shape_1, mode='nearest')
        att_0_1 = F.interpolate(atts[0], size=shape_1, mode='nearest') * atts[1]
        fea_2_1 = F.interpolate(feas[2], size=shape_1, mode='nearest')
        att_2_1 = F.interpolate(atts[2], size=shape_1, mode='nearest') * atts[1]
        fea_3_1 = F.interpolate(feas[3], size=shape_1, mode='nearest')
        att_3_1 = F.interpolate(atts[3], size=shape_1, mode='nearest') * atts[1]

        New_fea_1 = feas[1] * (1 + atts[1]) + (fea_0_1 * att_0_1) + (fea_2_1 * att_2_1) + (fea_3_1 * att_3_1)

        fea_0_2 = F.interpolate(feas[0], size=shape_2, mode='nearest')
        att_0_2 = F.interpolate(atts[0], size=shape_2, mode='nearest') * atts[2]
        fea_1_2 = F.interpolate(feas[1], size=shape_2, mode='nearest')
        att_1_2 = F.interpolate(atts[1], size=shape_2, mode='nearest') * atts[2]
        fea_3_2 = F.interpolate(feas[3], size=shape_2, mode='nearest')
        att_3_2 = F.interpolate(atts[3], size=shape_2, mode='nearest') * atts[2]

        New_fea_2 = feas[2] * (1 + atts[2]) + (fea_0_2 * att_0_2) + (fea_1_2 * att_1_2) + (fea_3_2 * att_3_2)

        fea_0_3 = F.interpolate(feas[0], size=shape_3, mode='nearest')
        att_0_3 = F.interpolate(atts[0], size=shape_3, mode='nearest') * atts[3]
        fea_1_3 = F.interpolate(feas[1], size=shape_3, mode='nearest')
        att_1_3 = F.interpolate(atts[1], size=shape_3, mode='nearest') * atts[3]
        fea_2_3 = F.interpolate(feas[2], size=shape_3, mode='nearest')
        att_2_3 = F.interpolate(atts[2], size=shape_3, mode='nearest') * atts[3]

        New_fea_3 = feas[3] * (1 + atts[3]) + (fea_0_3 * att_0_3) + (fea_1_3 * att_1_3) + (fea_2_3 * att_2_3)

        New_feas.append(New_fea_0)
        New_feas.append(New_fea_1)
        New_feas.append(New_fea_2)
        New_feas.append(New_fea_3)

        return New_feas


class SEModule(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(  #
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()  #
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = x * identity
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(self,
                 num_channels,
                 stride=1,
                 dw_size=3):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=num_channels,
            padding=1)
        self.dw_bn = nn.BatchNorm2d(
            num_features=num_channels)
        self.hardswish = nn.Hardswish()
        self.se = SEModule(num_channels)
        self.pw_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.pw_bn = nn.BatchNorm2d(
            num_features=num_channels)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.hardswish(x)
        x = self.se(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)

        return x


class ASPP(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling)
    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 2, 5, 1), fore_conv_kernel=(1, 3, 5, 1)):
        super().__init__()
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for index, dilation in enumerate(dilations):
            kernel_size = 3 if dilation > 1 else 1  # 1 3 3 1
            padding = dilation if dilation > 1 else 0  # 0 2 5 0
            fore_kernel = fore_conv_kernel[index]  # 1 3 5 1

            conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=fore_kernel, padding=int((fore_kernel - 1) / 2)),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=dilation,
                          padding=padding, bias=True))
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):  # (0, 1, 2, 3)
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


class CAM(nn.Module):
    def __init__(self, inplanes, reduction_ratio=1, fpn_lvl=4):  # inplanes = 256
        super(CAM, self).__init__()
        self.fpn_lvl = fpn_lvl  # 4
        self.dila_conv = nn.Sequential(nn.Conv2d(inplanes * fpn_lvl // reduction_ratio, inplanes // reduction_ratio,
                                                 kernel_size=3, stride=1, padding=1),
                                       ASPP(inplanes // reduction_ratio, inplanes // (4 * reduction_ratio)),
                                       nn.Conv2d(inplanes // reduction_ratio, inplanes // reduction_ratio,
                                                 kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(inplanes // reduction_ratio),
                                       nn.ReLU(inplace=False)
                                       )
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        self.upsample_cfg = dict(mode='nearest')
        self.down_conv = nn.ModuleList()
        self.att_conv = nn.ModuleList()
        self.down_conv_single = nn.ModuleList()
        self.se_conv_list = nn.ModuleList()
        for i in range(self.fpn_lvl):
            # change in_channel into 2 * inplanes
            self.att_conv.append(nn.Conv2d(2 * inplanes // reduction_ratio,
                                           1,
                                           kernel_size=3,
                                           stride=1,  # 2 ** i
                                           padding=1))
            #####
            if i == 0:
                down_stride = 1
            else:
                down_stride = 2
            self.down_conv.append(
                nn.Conv2d(inplanes // reduction_ratio, inplanes // reduction_ratio, kernel_size=3, stride=down_stride,
                          padding=1))
            # add defining single down_conv
            self.down_conv_single.append(
                nn.Conv2d(inplanes // reduction_ratio, inplanes // reduction_ratio, kernel_size=3, stride=2 ** i,
                          padding=1)
            )
            # add SEPD block
            self.se_conv_list.append(
                DepthwiseSeparable(inplanes // reduction_ratio)
            )
            ####

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):

        prev_shape = x[0].shape[2:]  # (w,h)
        multi_feats = [x[0]]

        for i in range(1, len(x)):
            pyr_feats_2x = F.interpolate(x[i], size=prev_shape, **self.upsample_cfg)
            multi_feats.append(pyr_feats_2x)

        multi_feats = torch.cat(multi_feats, 1)
        lvl_fea = self.dila_conv(multi_feats)
        #
        # add single down_conv
        atts_single = []
        for i in range(self.fpn_lvl):
            # 在这里先SE再卷积（并联*4）
            lvl_fea_single = self.down_conv_single[i](self.se_conv_list[i](lvl_fea))
            atts_single.append(lvl_fea_single)
        ####
        multi_atts = []
        for i in range(self.fpn_lvl):
            lvl_fea = self.down_conv[i](lvl_fea)
            # add cat feature
            concat_fea = torch.cat([lvl_fea, atts_single[i]], dim=1)
            ####
            lvl_att = self.att_conv[i](concat_fea)
            multi_atts.append(self.sigmoid(lvl_att))

        # visualization
        att_v = []
        for i in range(self.fpn_lvl):  # self.fpn_lvl
            att = F.interpolate(multi_atts[i], size=(765,1360), mode='bilinear')
            att_v.append(att)
            att = (att.detach().cpu().numpy()[0])
            # att /= np.max(att)
            #att = np.power(att, 0.8)
            att = att * 255
            att = att.astype(np.uint8).transpose(1, 2, 0)
            att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
            cv2.imwrite('results/sup{}.jpg'.format(i), att)
        #     cv2.waitKey(0)
                
        att_13 = att_v[0]*att_v[2]
        att = (att_13.detach().cpu().numpy()[0])
        att = att * 255
        att = att.astype(np.uint8).transpose(1, 2, 0)
        att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
        cv2.imwrite('results/sup13.jpg', att)
        
        att_14 = att_v[0]*att_v[3]
        att = (att_14.detach().cpu().numpy()[0])
        att = att * 255
        att = att.astype(np.uint8).transpose(1, 2, 0)
        att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
        cv2.imwrite('results/sup14.jpg', att)

        return multi_atts

@NECKS.register_module()
class SSFPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(SSFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.CAM = CAM(out_channels)
        self.build_new = build_new()
        # self.grads = {}
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.pan_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pan_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            out_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.pan_convs.append(pan_conv)
            self.out_convs.append(out_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):

        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build attention map
        att_list = self.CAM(laterals)

        laterals = [(1 + att_list[i]) * laterals[i] for i in range(len(laterals))]
        base_l_ls = []
        for i in range(4):
            base_l_ls.append(laterals[i])

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]

            # get intersection of Adjacent attention maps
            att_2x = F.interpolate(att_list[i], size=prev_shape, **self.upsample_cfg)
            att_insec = att_list[i - 1] * att_2x

            # get ROI of current attention map
            select_gate = att_insec

            laterals[i - 1] = laterals[i - 1] + select_gate * F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)
        laterals_fpn = [
            self.fpn_convs[i](laterals[i] + base_l_ls[i]) for i in range(used_backbone_levels)
        ]
        base_l_fpn = []
        for i in range(4):
            base_l_fpn.append(laterals_fpn[i])

        # build bottom-up path
        for i in range(1, 4):        # 1, 2, 3
            after_shape = laterals_fpn[i].shape[2:]

            att_half_x = F.interpolate(att_list[i-1], size=after_shape, **self.upsample_cfg)
            att_mul = att_list[i] * att_half_x

            save_gate = att_mul

            laterals_fpn[i] = laterals_fpn[i] + save_gate * F.interpolate(laterals_fpn[i-1], size=after_shape, **self.upsample_cfg)
        laterals_pan = [
            self.pan_convs[i](laterals_fpn[i] + base_l_fpn[i]) for i in range(used_backbone_levels)
        ]



        # build new
        laterals_new = self.build_new(laterals_pan, att_list)

        outs = [
            (1 + att_list[i]) * self.out_convs[i](laterals_new[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs), tuple(att_list)


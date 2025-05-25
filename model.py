# [核心文件] 模型定义文件：包含玉米南方锈病识别的UNet网络模型定义，实现图像分割和多任务学习（同时预测感染部位和感染等级）
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    UNet的基本卷积块：(Conv2d -> BatchNorm2d -> ReLU) × 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        初始化双卷积块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            mid_channels: 中间通道数，如果为None则等于out_channels
        """
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 第二个卷积层
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.double_conv(x)

class Down(nn.Module):
    """
    下采样模块: 最大池化 + 双卷积
    """
    def __init__(self, in_channels, out_channels):
        """
        初始化下采样模块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2x2最大池化，步长为2
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    上采样模块: 转置卷积/上采样 + 拼接 + 双卷积
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        初始化上采样模块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            bilinear: 是否使用双线性插值进行上采样，默认为True
        """
        super(Up, self).__init__()
        
        # 如果使用双线性插值，则使用普通卷积减少通道数
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 否则使用转置卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        前向传播
        
        参数:
            x1: 来自上一层的特征图
            x2: 来自对应下采样层的特征图（跳跃连接）
            
        返回:
            输出特征图
        """
        x1 = self.up(x1)
        
        # 对齐尺寸，防止因为输入图像尺寸不是2的幂次导致的特征图尺寸不匹配
        # [N, C, H, W]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # 使用填充使尺寸匹配
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # 拼接特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    输出卷积层：将通道数映射到所需的输出通道数
    """
    def __init__(self, in_channels, out_channels):
        """
        初始化输出卷积层
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        return self.conv(x)

class ChannelAttention(nn.Module):
    """
    通道注意力机制
    捕捉通道之间的依赖关系，对重要的通道赋予更高的权重
    结合平均池化和最大池化的信息，提高特征表示能力
    """
    def __init__(self, in_channels, reduction_ratio=16):
        """
        初始化通道注意力模块
        
        参数:
            in_channels: 输入特征图的通道数
            reduction_ratio: 降维比例，用于减少参数量
        """
        super(ChannelAttention, self).__init__()
        # 全局平均池化 - 捕获通道的全局分布
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出1x1特征图
        # 全局最大池化 - 捕获通道的显著特征
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 通过两个1x1卷积实现全连接层，减少参数量
        self.fc = nn.Sequential(
            # 第一个1x1卷积，降维
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            # 第二个1x1卷积，恢复维度
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        # sigmoid激活函数，将注意力权重归一化到[0,1]范围
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征图，形状为 [batch_size, in_channels, height, width]
            
        返回:
            attention: 通道注意力权重，形状为 [batch_size, in_channels, 1, 1]
        """
        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x))
        # 最大池化分支
        max_out = self.fc(self.max_pool(x))
        # 融合两个分支的信息
        out = avg_out + max_out
        # 应用sigmoid归一化
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    空间注意力机制
    关注图像的空间位置重要性，对重要区域赋予更高权重
    结合通道平均值和最大值的信息，增强模型对空间区域的感知能力
    """
    def __init__(self, kernel_size=7):
        """
        初始化空间注意力模块
        
        参数:
            kernel_size: 卷积核大小，默认为7，用于捕获更大的感受野
        """
        super(SpatialAttention, self).__init__()
        # 使用单层卷积学习空间注意力图
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()  # 注意力权重归一化

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征图，形状为 [batch_size, channels, height, width]
            
        返回:
            attention: 空间注意力权重，形状为 [batch_size, 1, height, width]
        """
        # 沿通道维度计算平均值 - 捕获全局通道信息
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 沿通道维度计算最大值 - 捕获显著特征
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接通道平均值和最大值
        x = torch.cat([avg_out, max_out], dim=1)  # 形状为 [batch_size, 2, height, width]
        # 通过卷积生成空间注意力图
        x = self.conv(x)  # 输出单通道特征图
        # 应用sigmoid归一化
        return self.sigmoid(x)

class AttentionBlock(nn.Module):
    """
    注意力模块: 结合通道注意力和空间注意力
    """
    def __init__(self, in_channels):
        """
        初始化注意力模块
        
        参数:
            in_channels: 输入通道数
        """
        super(AttentionBlock, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征图
            
        返回:
            增强后的特征图
        """
        x = self.ca(x) * x  # 应用通道注意力
        x = self.sa(x) * x  # 应用空间注意力
        return x

class UNet(nn.Module):
    """
    UNet网络: 用于玉米南方锈病图像分割与多任务学习
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], img_size=128, bilinear=True, with_attention=True):
        """
        初始化UNet网络
        
        参数:
            in_channels: 输入通道数，默认为3（RGB）
            out_channels: 分割输出通道数，默认为1（二分类）
            features: 各层特征通道数，默认为[64, 128, 256, 512]
            img_size: 输入图像尺寸，默认为128x128
            bilinear: 是否使用双线性插值进行上采样，默认为True
            with_attention: 是否使用注意力机制，默认为True
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.bilinear = bilinear
        self.with_attention = with_attention
        
        # 初始双卷积层
        self.inc = DoubleConv(in_channels, features[0])
        
        # 下采样层
        self.downs = nn.ModuleList()
        in_features = features[0]
        for feature in features[1:]:
            self.downs.append(Down(in_features, feature))
            in_features = feature
        
        # 上采样层
        self.ups = nn.ModuleList()
        features = features[::-1]  # 反转特征列表
        for i in range(len(features) - 1):
            if i == 0:
                # 第一个上采样层的输入通道数等于瓶颈层特征数
                self.ups.append(Up(features[i], features[i+1], bilinear=bilinear))
            else:
                # 其他上采样层的输入通道数为当前特征数
                self.ups.append(Up(features[i], features[i+1], bilinear=bilinear))
        
        # 分割输出层
        self.outc = OutConv(features[-1], out_channels)
        
        # 注意力模块
        if with_attention:
            self.attention_blocks = nn.ModuleList()
            for feature in features:
                self.attention_blocks.append(AttentionBlock(feature))
        
        # 计算卷积后的特征图大小
        # 经过所有下采样(2^4=16)，尺寸变为原来的1/16
        conv_output_size = img_size // 16
        bottleneck_features = features[0]
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 位置分类头 (3分类) - 将特征向量映射到3个类别
        self.position_classifier = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为一维向量
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(bottleneck_features, 128),  # 全连接层降维
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),  # 第二个Dropout层进一步防止过拟合
            nn.Linear(128, 3)  # 输出3个类别的logits: 下部/中部/上部
        )
        
        # 等级分类头 (改为回归任务) - 将特征向量映射到1个输出值（回归）
        self.grade_classifier = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为一维向量
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(bottleneck_features, 128),  # 全连接层降维
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # 输出1个值，用于回归预测感染等级
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像 [batch_size, in_channels, height, width]
            
        返回:
            tuple: (segmentation, position_logits, grade_values)
                - segmentation: 分割输出 [batch_size, out_channels, height, width]
                - position_logits: 位置分类输出 [batch_size, 3]
                - grade_values: 等级回归输出 [batch_size, 1]
        """
        # 编码器部分
        x1 = self.inc(x)  # 初始特征提取
        
        # 存储下采样中间结果，用于跳跃连接
        features = [x1]
        
        # 下采样路径
        for i, down in enumerate(self.downs):
            features.append(down(features[-1]))
        
        # 瓶颈层特征
        bottleneck = features[-1]
        
        # 应用注意力机制到瓶颈层
        if self.with_attention:
            bottleneck = self.attention_blocks[0](bottleneck)
        
        # 从瓶颈层提取多任务特征
        task_features = self.avgpool(bottleneck)
        
        # 位置分类
        position_logits = self.position_classifier(task_features)
        
        # 等级分类
        grade_values = self.grade_classifier(task_features)
        
        # 上采样路径
        x = bottleneck
        
        # 反向遍历所有特征图，除了第一个
        for i, up in enumerate(self.ups):
            # 跳跃连接，使用存储的特征图
            skip_feature = features[-(i+2)]
            
            # 应用注意力机制到跳跃连接的特征图
            if self.with_attention:
                skip_feature = self.attention_blocks[i+1](skip_feature)
            
            x = up(x, skip_feature)
        
        # 分割输出
        segmentation = self.outc(x)
        
        return segmentation, position_logits, grade_values

class UNetPlusPlus(nn.Module):
    """
    UNet++网络: 比UNet有更密集的跳跃连接，进一步提高分割效果
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], img_size=128, bilinear=True, with_attention=True):
        """
        初始化UNet++网络
        
        参数:
            in_channels: 输入通道数，默认为3（RGB）
            out_channels: 分割输出通道数，默认为1（二分类）
            features: 各层特征通道数，默认为[64, 128, 256, 512]
            img_size: 输入图像尺寸，默认为128x128
            bilinear: 是否使用双线性插值进行上采样，默认为True
            with_attention: 是否使用注意力机制，默认为True
        """
        super(UNetPlusPlus, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.bilinear = bilinear
        self.with_attention = with_attention
        
        self.n_classes = out_channels
        self.depth = len(features)
        
        # 初始双卷积层
        self.inc = DoubleConv(in_channels, features[0])
        
        # 下采样层
        self.downs = nn.ModuleList()
        in_features = features[0]
        for feature in features[1:]:
            self.downs.append(Down(in_features, feature))
            in_features = feature
        
        # 创建UNet++的嵌套结构
        # 初始化空的嵌套结构以保存中间层
        self.nested_conv_blocks = nn.ModuleList()
        
        # 创建UNet++密集跳跃连接的嵌套结构
        for i in range(self.depth):
            nested_channels = []
            for j in range(self.depth - i):
                if j == 0:
                    # 第一列已由encoder/decoder初始化
                    in_ch = features[i]
                    out_ch = features[i]
                else:
                    # 计算输入通道: 前一列的输出通道 + 上一行的输出通道
                    # UNet++中，输入通道数是所有前面连接的通道之和
                    in_ch = features[i] + features[i+1]
                    out_ch = features[i]
                
                # 每个位置都是一个双卷积块
                conv_block = DoubleConv(in_ch, out_ch)
                nested_channels.append(conv_block)
            
            self.nested_conv_blocks.append(nn.ModuleList(nested_channels))
        
        # 上采样层 - 用于连接不同深度的特征
        self.up_blocks = nn.ModuleList()
        for i in range(self.depth - 1):
            for j in range(self.depth - i - 1):
                # 上采样操作 - 从更深层到浅层
                if bilinear:
                    up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    up = nn.ConvTranspose2d(features[i+j+1], features[i+j+1], kernel_size=2, stride=2)
                self.up_blocks.append(up)
        
        # 分割输出层
        self.outc = OutConv(features[0], out_channels)
        
        # 注意力模块
        if with_attention:
            self.attention_blocks = nn.ModuleList()
            for feature in features:
                self.attention_blocks.append(AttentionBlock(feature))
        
        # 计算卷积后的特征图大小
        # 经过所有下采样(2^depth=16)，尺寸变为原来的1/16
        conv_output_size = img_size // (2 ** (self.depth - 1))
        bottleneck_features = features[-1]
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 位置分类头 (3分类) - 将特征向量映射到3个类别
        self.position_classifier = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为一维向量
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(bottleneck_features, 128),  # 全连接层降维
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),  # 第二个Dropout层进一步防止过拟合
            nn.Linear(128, 3)  # 输出3个类别的logits: 下部/中部/上部
        )
        
        # 等级分类头 (改为回归任务) - 将特征向量映射到1个输出值（回归）
        self.grade_classifier = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为一维向量
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(bottleneck_features, 128),  # 全连接层降维
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # 输出1个值，用于回归预测感染等级
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像 [batch_size, in_channels, height, width]
            
        返回:
            tuple: (segmentation, position_logits, grade_values)
                - segmentation: 分割输出 [batch_size, out_channels, height, width]
                - position_logits: 位置分类输出 [batch_size, 3]
                - grade_values: 等级回归输出 [batch_size, 1]
        """
        # 存储所有中间特征图
        encoders = []
        
        # 第一层特征
        x0_0 = self.inc(x)
        encoders.append(x0_0)
        
        # 编码器路径
        for i, down in enumerate(self.downs):
            if i == 0:
                x1_0 = down(x0_0)
            elif i == 1:
                x2_0 = down(x1_0)
            elif i == 2:
                x3_0 = down(x2_0)
            elif i == 3:
                x4_0 = down(x3_0)
            encoders.append(locals()[f'x{i+1}_0'])
        
        # 瓶颈层特征
        bottleneck = encoders[-1]
        
        # 应用注意力机制到瓶颈层
        if self.with_attention:
            bottleneck = self.attention_blocks[0](bottleneck)
        
        # 从瓶颈层提取多任务特征
        task_features = self.avgpool(bottleneck)
        
        # 位置分类
        position_logits = self.position_classifier(task_features)
        
        # 等级分类
        grade_values = self.grade_classifier(task_features)
        
        # UNet++ 密集嵌套结构
        # 初始化用于存储每个节点的特征
        nested_features = {}
        up_idx = 0
        
        # 第一列已在编码过程中计算
        for i in range(self.depth):
            nested_features[(i, 0)] = encoders[i]
        
        # 计算嵌套UNet的节点
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                # 拼接特征图
                prev_feat = nested_features[(i, j-1)]
                
                # 当前层的上采样层
                up = self.up_blocks[up_idx]
                up_idx += 1
                
                # 上采样上一层的特征
                up_feat = up(nested_features[(i+1, j-1)])
                
                # 裁剪和拼接
                diffY = prev_feat.size()[2] - up_feat.size()[2]
                diffX = prev_feat.size()[3] - up_feat.size()[3]
                
                up_feat = F.pad(up_feat, [diffX // 2, diffX - diffX // 2,
                                         diffY // 2, diffY - diffY // 2])
                
                # 拼接操作
                cat_feat = torch.cat([prev_feat, up_feat], dim=1)
                
                # 应用注意力机制
                if self.with_attention:
                    cat_feat = self.attention_blocks[j](cat_feat)
                
                # 双卷积处理
                nested_features[(i, j)] = self.nested_conv_blocks[i][j](cat_feat)
        
        # 输出层 - 使用最上层特征图进行分割
        segmentation = self.outc(nested_features[(0, self.depth-1)])
        
        return segmentation, position_logits, grade_values

def get_model(model_type='unet', in_channels=3, out_channels=1, img_size=128, bilinear=True, with_attention=True, features=None):
    """
    获取模型实例
    
    参数:
        model_type: 模型类型，选项为:
            - 'unet': 标准UNet模型
            - 'unet++': UNet++模型，增强版UNet
        in_channels: 输入通道数
        out_channels: 分割输出通道数
        img_size: 输入图像尺寸
        bilinear: 是否使用双线性插值
        with_attention: 是否使用注意力机制
        features: 特征通道列表，默认为[64, 128, 256, 512]
        
    返回:
        model: 模型实例
        
    异常:
        ValueError: 当model_type不是支持的类型时抛出
    """
    if features is None:
        features = [64, 128, 256, 512]
    
    if model_type == 'unet':
        return UNet(in_channels=in_channels, out_channels=out_channels, features=features, 
                    img_size=img_size, bilinear=bilinear, with_attention=with_attention)
    elif model_type == 'unet++':
        return UNetPlusPlus(in_channels=in_channels, out_channels=out_channels, features=features, 
                           img_size=img_size, bilinear=bilinear, with_attention=with_attention)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
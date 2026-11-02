# 技术设计文档（Design Doc）：端侧混合重打光与参考图光影复刻管线

## 基于会前 Greenroom 离线烘焙（Pre-Baking）、60fps 球面商贴图（Quotient MatCap）与 Multi-Agent 协同开发的完整工程方案

Author: Chenchen Tang (cctang@google.com) \| Date: 2026-10-26 \| Status: Draft / Architecture Proposal \| Target: On-Device Real-Time Video (WebGL / OpenGL ES / Metal) & Multi-Agent Implementation

核心设计结论（Executive Summary）：本方案旨在打破普通用户在直播、视频会议（如 Google Meet）与电商静物拍摄中的“摄影布光与调色认知鸿沟（Knowledge Gap）”。通过将“单图逆向物理测光”、“会前 Greenroom 1.5 秒离线外观烘焙（Offline Quotient MatCap Baking）”、“会中 \<0.5ms 极速法线查表渲染”以及“拍后 9 宫格肤色保护仿色与美学排版”融为一体，系统可在零神经网络会中推理开销下实现 1080p/4K @ 60fps 的电影级零闪烁重打光，达到专业影棚实拍 92%\~96% 的视觉还原度。

# 1\. 背景、行业痛点与破局定位（Context & Problem Statement）

在大模型（Video Diffusion / DiT）与传统图形学（CG）交汇的当下，实时视频编辑与人像/商品重打光（Relighting）长期处于“算力墙”与“物理真实感”的两难困境中：

* 生成式扩散模型派（如 IC-Light、Neural Gaffer、LiveLight）：具备极佳的复杂阴影与高光生成能力，但受限于多步迭代去噪与 VAE 编解码损耗。例如 Neural Gaffer 在 A6000 上单图（256×256）需 0.8\~1.5s；LiveLight 虽压缩至 4 步去噪（15.78 FPS），但端到端延迟高达 253ms 且需维护高显存滚动窗口阶梯（Denoising Ladder），移动端无法实时运行且易丢失 4K 原生毛孔细节。  
* 端侧传统实时管线派（如 Google Meet 现有 Studio Lighting、EdgeRelight360）：Google Meet 现有方案在会议进行中逐帧/隔帧运行 Segmentation、Pixel-wise Normals 与 Albedo 神经网络，导致硬件门槛极高（要求 8 核/12 线程如 i9 或 M1 以上，低配机被迫回退云端），且仅支持基础 Lambertian 漫反射；EdgeRelight360 则仅做“加法补光（Light-Adding）”与简单法线时序均值滤波，无法擦除原图顶灯死黑阴影，快速转头时还会产生光影拖影（Ghosting）。  
* 用户侧操作鸿沟（Knowledge Gap）：现有商业修图软件（如 Evoto AI、醒图、Luminar Neo）要求用户手动盲调 3D 虚拟灯泡滑块；而硬件灯控软件（如爱图仕 Sidus Link Pro、set.a.light 3D）仅面向专业灯光师提供正向控制，缺乏从“一张好看的参考图”逆向反推物理灯位与居家替代方案的能力。

## 1.1 竞品格局与本方案的差异化切入点（Wedge Strategy）

&nbsp;

| 赛道维度 | 代表竞品 / 方案 | 核心能力与现状 | 致命短板（本方案破局点） |
| :---- | :---- | :---- | :---- |
| 实时视频会议打光 | Google Meet (Studio Lighting) | 会中实时跑法线与 Albedo 网络 \+ WebGL Lambertian 着色器。 | 会中推理开销极大（需 8 核\+ CPU）；无离线烘焙、鼻翼阴影与 SSS。 |
| 商业人像精修与单图打光 | Evoto AI、Focos、Luminar Neo | 支持高保真毛孔保留、去眼镜反光及手动 3D 打光调整。 | 纯后期修图，需手动盲调虚拟光源角度，无“参考图一键逆向复刻”与物理摆灯指导。 |
| 影视灯光硬件智控 | Sidus Link Pro (爱图仕)、set.a.light 3D | 垄断蓝牙 Mesh 灯控协议，支持 3D 布光预演与导出 2D 机位图。 | 纯正向工具，无 CV 逆向视觉测光能力；用户缺灯时无法用端侧 Shader 虚拟补齐。 |
| 参考图仿色与组图排版 | Retouch4Me、Color.io、Polarr | 支持单图 3D LUT 提取及带肤色保护的色彩迁移。 | 偏向 PC 专业流程，缺乏移动端“肤色锁仿色 \+ 朋友圈 9 宫格视觉重心自动排序”闭环。 |

# 2. 系统核心架构设计（System Architecture）

本系统采用“物理教练 \+ 软硬混合渲染（Neuro-Symbolic Hybrid CG Pipeline）”架构，全面覆盖“拍前逆向指导 → 会前 1.5s 离线烘焙 → 会中 60fps 极速渲染 → 拍后 9 图色彩统一”四大阶段：

## 2.1 拍前逆向测光：从单张参考图反推物理摄影棚灯位（Inverse Studio Lighting）

利用人脸或物体的几何与物理反射线索，系统将连续环境光场逆向拟合为最多 3 盏离散物理灯具（主光 Key、辅光 Fill、轮廓光 Rim）加环境底光（Ambient）：

* 线索 1 — 漫反射明暗与主辅光比（Diffuse Shading）：通过 MediaPipe Face Mesh（\<3ms 获取 468 个 3D 顶点）或通用物体法线网络 DSINE 计算表面法线 N(p)，提取纯光照层 S(p)。左右脸颊/物体亮暗面比值直接映射为主辅光比（Key-to-Fill Ratio，如 3.2:1 立体伦勃朗光 vs 1:1 平光）。  
* 线索 2 — 投射阴影与柔光罩尺寸（Cast Shadow & Penumbra）：鼻底蝴蝶影对应 12 点钟高位光，远光侧三角亮区对应 45° 伦勃朗光；通过 Sobel 算子测量阴影边缘半影（Penumbra）过渡像素宽度，反推光源柔光度（Softness：刀割硬边 \= 裸灯，宽渐变 \= 60cm+ 八角柔光箱）。  
* 线索 3 — 瞳孔眼神光与高光脊线（Catchlights & Specular Ridge）：将人眼角膜视为天然凸面镜（Chrome Ball），或对高反光金属物体直接应用视线反射定律 L(p) \= 2(N(p)·V)N(p) \- V，反推灯具精确形状（环形灯 vs 条形柔光箱）与空间方位。  
* 线索 4 — 发丝/边缘亮边（Rim Halo）：提取前景分割掩码内侧 5\~10px 环形边界（Ring Boundary），检测高亮度与色温偏移，锁定后方轮廓灯（Kicker/Rim Light）的方位与 RGB 色彩。

参数化光源拟合方程（Parametric Studio Light Solver）：已知像素法线 N(p) 与观测光照 S(p)，在球面预设 M=64 个候选方向网格，通过非负最小二乘（NNLS）或 L-BFGS 求解以下稀疏叠加模型（耗时 \<5ms）：

S(p) ≈ I\_ambient \+ Σ\_{k=1..3} \[ c\_k \* I\_k \* max(N(p) · ω(θ\_k, φ\_k), 0)^(γ\_k) \]

## 2.2 扩展至任意物体打光（Universal Object Relighting Extension）

当目标从“人脸”扩展至“电商商品、静物美食、手办汽车”时，系统失去固定五官网格与固定皮肤粗糙度先验，通过以下三大模块平滑升级：

&nbsp;

| 材质类型 (基于 Roughness/透明度路由) | 代表商品场景 | 逆向测光核心锚点 | 物理摆灯指导与渲染策略 |
| :---- | :---- | :---- | :---- |
| 类型 A：漫反射哑光物体 (Roughness \> 0.5) | 毛绒玩具、烘焙美食、木制品、哑光包装盒 | DSINE 通用法线 N \+ 桌面投射阴影方向与半影宽度 | 遵循漫反射 N·L 定律；指导用户采用 45° 柔光主灯 \+ 对侧反光板补光。 |
| 类型 B：高光/镜面金属物体 (Roughness \< 0.2) | 金属腕表、口红管、耳机、易拉罐、汽车模型 | 表面高光亮条反射追踪 L \= 2(N·V)N \- V（亮条即灯箱倒影） | 切忌裸灯直闪！指导紧贴侧上方 45° 放置大型条形柔光箱或隔硫酸纸打渐变高光腰线。 |
| 类型 C：透明玻璃/液体材质 | 香水瓶、红酒杯、玻璃护肤品瓶身 | TransNormal-2 透明物体专用法线 \+ 瓶身边缘黑白轮廓线检测 | 正面不打光！指导采用“亮调黑线法（12点钟背光打亮背景+两侧黑吸光板）”或“暗调亮线法”。 |

## 2.3 会前 Greenroom 离线烘焙 \+ 会中 0.5ms 球面商贴图（Pre-Baking & Quotient MatCap）

针对视频会议（Google Meet / Zoom）与固定机位直播中“房间物理光源恒定不变（Static Room Illumination）”的核心特性，本设计摒弃了会中逐帧运行逆渲染神经网络的沉重做法，首创“会前离线烘焙球面商贴图（Spherical Quotient MatCap）”机制：

* 为什么不能固定在 2D 屏幕坐标 (x, y)？若在屏幕 2D 空间固定提亮/压暗贴图，一旦用户在会中歪头或转身，面部几何与屏幕固定光斑就会发生脱节。  
* 为什么必须固定在 3D 表面法线空间 (Nx, Ny, Nz)？当房间光源固定时，无论人脸如何旋转，只要某处皮肤表面法线转到特定方向（例如朝左 45°），该微表面接收到的房间环境光 L\_room(N) 与目标虚拟光 L\_target(N) 的比值永远是恒定的（忽略微小位移）。  
* 阶段 A（会前 Greenroom 预检离线烘焙，耗时 \~1.5s）：用户在进入会议前的预览页停留并微转头 1 秒，系统运行高精度本征分解（如 SPLiT / 球谐拟合）结合多视角光度立体法（Photometric Stereo），精准解耦出房间环境光场 L\_room(N)、个人专属皮肤基底 Albedo 及颜色校正矩阵 CCM，并将“去原光（De-light）+ 加新光（Add-light）+ 预积分次表面散射（Pre-Integrated SSS）”一次性烘焙为一张 64×64 像素的球面比值查找表 Q\_baked(Nx, Ny) 以及 3 张 Canonical UV 鼻翼软阴影贴图。  
* 阶段 B（会中 Online 60fps 极速查表，耗时 \<0.5ms/帧）：会议全程零神经网络重度推理！每帧仅由轻量关键点插值出顶点法线 N(x,y)，转换至 MatCap UV 坐标 (Nx\*0.5+0.5, Ny\*0.5+0.5) 对 64×64 查找表做 1 次 texture2D 采样，并乘回 1080p/4K 原生画面（商图像 Quotient Image 机制），100% 保留原生 4K 毛孔与胡茬且数学级零闪烁。  
* 阶段 C（0.2fps 低频后台心跳容错）：后台协程每 5 秒抽检 1 帧（或当检测到全图亮度因切换全白 PPT 突变 \>15% 时），异步耗时 50ms 重新烘焙 64×64 MatCap 并通过 1 秒 Cross-fade 平滑替换，兼顾零功耗与环境光突变自适应。

&nbsp;

| 子任务模块 | 会前 Offline 阶段 (算 1 次，耗时 \~1.5s) | 会中 Online 阶段 (60fps 循环，耗时 \<0.5ms) | 端侧算力与画质收益 |
| :---- | :---- | :---- | :---- |
| 1\. 房间原光去除 (De-lighting) | 运行完整逆渲染模型，拟合房间 9 阶球谐光场并写入预烘焙分母。 | 0 独立计算（已融合在 Quotient MatCap 分母中）。 | 省掉 100% 会中逆渲染网络开销，消除顶灯黑眼圈。 |
| 2\. 目标光合成与皮肤 SSS | 将目标光场与 Pre-Integrated SSS 积分，烘焙为 64×64 Quotient\_MatCap。 | 每像素仅用 (Nx, Ny) 做 1 次 texture2D 采样并乘回 4K 原图。 | 保留 100% 4K 毛孔细节，彻底消除塑料石膏脸。 |
| 3\. 鼻翼/眼窝投射阴影 (Cast Shadow) | 在标准正脸 Canonical UV 空间针对左/中/右 3 个朝向离线光线追踪生成高清软阴影贴图。 | 根据实时头部旋转角 (Yaw/Pitch) 对 3 张 UV 阴影贴图做简单线性插值 (Lerp)。 | 免除逐帧 Ray-Marching，实时呈现柔美伦勃朗三角鼻影。 |
| 4\. 眼神光与菲涅尔轮廓光 | 解析参考图 Catchlight 形状（环灯/方箱）与轮廓光色温。 | 基于虹膜 3D 球心反射角动态叠加眼神光 \+ 边缘 (1 \- N·V)^4 菲涅尔轮廓光。 | 转头时眼神光灵动跟随，发丝与肩部轮廓立体通透。 |

# 3. 核心算法与 Shader 实现规格（Core Shaders & Algorithms）

## 3.1 会中 60fps 极速重打光 Fragment Shader（Quotient MatCap \+ Cast Shadow \+ PBR）

以下为运行在 WebGL2 / OpenGL ES 3.0 / Metal 管线中的核心会中/直播实时着色器代码，包含预烘焙球面商查表、UV 投射鼻影、微表面高光及菲涅尔边缘光：

\#version 300 es

precision highp float;

&nbsp;

in vec2 v\_uv;

in vec2 v\_face\_canonical\_uv;

in vec3 v\_normal;          // 由轻量 FaceMesh 顶点法线经光流 EMA 平滑后插值得到

in float v\_face\_alpha;     // 人像前景与皮肤平滑掩码

&nbsp;

uniform sampler2D u\_video\_frame;            // 1080p/4K 原生高清视频帧

uniform sampler2D u\_baked\_quotient\_matcap;  // 64x64 会前预烘焙球面商贴图 (TargetLight\_SSS / RoomLight)

uniform sampler2D u\_baked\_uv\_shadow;        // 预烘焙 Canonical UV 鼻翼投射软阴影贴图

uniform mat3 u\_ccm\_matrix;                  // 会前离线校准的 3x3 颜色与白平衡校正矩阵

uniform vec3 u\_virtual\_key\_dir;             // 主光 3D 单位向量 (用于动态高光与眼神光)

uniform vec3 u\_rim\_light\_color;             // 轮廓光 RGB 颜色与强度

uniform float u\_roughness;                  // 表面粗糙度 (人脸默认 \~0.5，金属商品 \<0.2)

&nbsp;

out vec4 fragColor;

&nbsp;

void main() {

&nbsp;&nbsp;&nbsp;&nbsp;vec3 orig\_rgb \= texture(u\_video\_frame, v\_uv).rgb;

&nbsp;&nbsp;&nbsp;&nbsp;vec3 N \= normalize(v\_normal);

&nbsp;&nbsp;&nbsp;&nbsp;vec3 V \= vec3(0.0, 0.0, 1.0);

&nbsp;&nbsp;&nbsp;&nbsp;vec3 L \= normalize(u\_virtual\_key\_dir);

&nbsp;&nbsp;&nbsp;&nbsp;vec3 H \= normalize(L \+ V);

&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;// 1\. 标准 MatCap 球面映射: 将法线 (Nx, Ny ∈ \[-1, 1\]) 映射至 \[0, 1\] UV

&nbsp;&nbsp;&nbsp;&nbsp;vec2 matcap\_uv \= N.xy \* 0.5 \+ 0.5;

&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;// 2\. 仅 1 次纹理采样！同时完成 De-light(除原光) \+ Add-light(加新光) \+ 皮肤 SSS 红润过渡

&nbsp;&nbsp;&nbsp;&nbsp;vec3 light\_ratio \= texture(u\_baked\_quotient\_matcap, matcap\_uv).rgb;

&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;// 3\. 采样 Canonical UV 鼻翼投射阴影 (若为非预烘焙模式，可替换为 8-step 2.5D Depth Ray-Marching)

&nbsp;&nbsp;&nbsp;&nbsp;float cast\_shadow \= texture(u\_baked\_uv\_shadow, v\_face\_canonical\_uv).r;

&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;// 4\. 微表面镜面高光 (Blinn-Phong PBR) 与菲涅尔边缘轮廓光 (Fresnel Rim Light)

&nbsp;&nbsp;&nbsp;&nbsp;float shininess \= pow(2.0, (1.0 \- u\_roughness) \* 10.0);

&nbsp;&nbsp;&nbsp;&nbsp;float spec \= pow(max(dot(N, H), 0.0), shininess) \* (1.0 \- u\_roughness);

&nbsp;&nbsp;&nbsp;&nbsp;float fresnel \= pow(1.0 \- max(dot(N, V), 0.0), 4.0);

&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;// 5\. 商图像乘法合成: 100% 无损保留 4K 原生毛孔与纹理细节

&nbsp;&nbsp;&nbsp;&nbsp;vec3 calibrated\_rgb \= u\_ccm\_matrix \* orig\_rgb;

&nbsp;&nbsp;&nbsp;&nbsp;vec3 relit\_face \= calibrated\_rgb \* (light\_ratio \* cast\_shadow) \+ spec \* 0.15 \+ fresnel \* u\_rim\_light\_color;

&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;vec3 final\_rgb \= mix(orig\_rgb, relit\_face, v\_face\_alpha);

&nbsp;&nbsp;&nbsp;&nbsp;fragColor \= vec4(clamp(final\_rgb, 0.0, 1.0), 1.0);

}

## 3.2 端侧视频编辑跨帧光流补全算子（Zero-Weight Pyramid Flow Completion）

当管线用于离线视频局部重绘（Video Inpainting）或跨帧法线/掩码传播时，为消除逐帧独立生成导致的高频闪烁（Flickering）与纹理粘滞（Texture Sticking），系统采用“无神经网络权重的金字塔拉普拉斯光流扩散算子”，在 1/8 分辨率下仅靠基础池化算子即可在 0.2ms 内补全被遮挡区域的背景光流，并执行源头单次拉取（One-Shot Backward Pulling）：

def complete\_flow\_pyramid(flow: torch.Tensor, mask: torch.Tensor) \-\> torch.Tensor:

&nbsp;&nbsp;&nbsp;&nbsp;orig\_H, orig\_W \= flow.shape\[-2:\]

&nbsp;&nbsp;&nbsp;&nbsp;\# Step 0: 形态学膨胀 9px，吃掉遮挡物边缘溢出的前景运动光流 (防 Foreground Motion Leakage)

&nbsp;&nbsp;&nbsp;&nbsp;dilated\_mask \= F.max\_pool2d(mask, kernel\_size=9, stride=1, padding=4)

&nbsp;&nbsp;&nbsp;&nbsp;\# Step 1: 降采样至 1/8 分辨率 (面积缩小 64 倍，位移数值同步乘 0.125)

&nbsp;&nbsp;&nbsp;&nbsp;f\_low \= F.interpolate(flow, scale\_factor=0.125, mode='bilinear', align\_corners=False) \* 0.125

&nbsp;&nbsp;&nbsp;&nbsp;m\_low \= F.interpolate(dilated\_mask, scale\_factor=0.125, mode='nearest')

&nbsp;&nbsp;&nbsp;&nbsp;valid \= 1.0 \- m\_low

&nbsp;&nbsp;&nbsp;&nbsp;\# Step 2: 归一化卷积初始化洞心 \+ 8 次 3x3 雅可比迭代 (求解离散拉普拉斯方程 ∇²u=0, ∇²v=0)

&nbsp;&nbsp;&nbsp;&nbsp;init\_flow \= F.avg\_pool2d(f\_low \* valid, 15, 1, 7\) / (F.avg\_pool2d(valid, 15, 1, 7\) \+ 1e-6)

&nbsp;&nbsp;&nbsp;&nbsp;curr\_flow \= valid \* f\_low \+ (1.0 \- valid) \* init\_flow

&nbsp;&nbsp;&nbsp;&nbsp;for \_ in range(8):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;smoothed \= F.avg\_pool2d(curr\_flow, kernel\_size=3, stride=1, padding=1)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;curr\_flow \= valid \* f\_low \+ (1.0 \- valid) \* smoothed

&nbsp;&nbsp;&nbsp;&nbsp;\# Step 3: 双线性上采样回原尺寸，数值乘回 8.0

&nbsp;&nbsp;&nbsp;&nbsp;completed \= F.interpolate(curr\_flow, size=(orig\_H, orig\_W), mode='bilinear', align\_corners=False) \* 8.0

&nbsp;&nbsp;&nbsp;&nbsp;return (1.0 \- dilated\_mask) \* flow \+ dilated\_mask \* completed

## 3.3 拍后扩展：带肤色保护锁的朋友圈 9 图色调统一与美学排版算法

针对日常多图发圈场景，传统全局 Reinhard 仿色易导致人脸被环境色污染成“橘子人/阿凡达脸”，且不同场景照片强套同一 LUT 会导致方差坍塌。本系统采用“本征分离 \+ 肤色锁（Skin-Albedo Lock）+ 九宫格组合优化器”：

// WebGL 9-Grid Skin-Protected Color Transfer 核心逻辑

vec3 lab\_curr \= rgb2lab(orig\_rgb);

// 1\. 自适应阻尼方差对齐 (防止跨度极大的室内外照片过曝或死黑)

vec3 scale\_std \= clamp(u\_ref\_std / max(u\_curr\_std, vec3(1e-4)), vec3(0.6), vec3(1.6));

vec3 lab\_matched \= (lab\_curr \- u\_curr\_mean) \* scale\_std \+ u\_ref\_mean;

&nbsp;

// 2\. 人像肤色保护锁 (u\_skin\_mask: 1.0 为面部/身体皮肤, 0.0 为背景)

// 皮肤区域保留 80% 原始健康色相 (ab 通道)，同时迁移 70% 参考图的立体明暗反差 (L 通道)

vec3 lab\_final;

lab\_final.x  \= mix(lab\_matched.x,  mix(lab\_curr.x, lab\_matched.x, 0.7), u\_skin\_mask);

lab\_final.yz \= mix(lab\_matched.yz, mix(lab\_matched.yz, lab\_curr.yz, 0.8), u\_skin\_mask);

vec3 final\_rgb \= lab2rgb(lab\_final);

# 4\. Multi-Agent 并行开发协同规范与接口契约（Multi-Agent Execution Plan）

为支持 5 个 AI 编码 Agent（如 Claude Code / Cursor Agents）在零阻塞条件下同时并行开发，工程采用 Monorepo 架构，通过冻结顶层 JSON 与 GPU 纹理契约（/packages/shared-schema）实现彻底解耦。

## 4.1 全局冻结数据契约（target\_profile.schema.json）

{

&nbsp;&nbsp;"subject\_type": "portrait \| object\_diffuse \| object\_specular \| object\_transparent",

&nbsp;&nbsp;"lighting\_rig": {

&nbsp;&nbsp;&nbsp;&nbsp;"ambient\_rgb": \[0.15, 0.15, 0.18\],

&nbsp;&nbsp;&nbsp;&nbsp;"key\_to\_fill\_ratio": 3.2,

&nbsp;&nbsp;&nbsp;&nbsp;"lights": \[

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"id": "key\_light",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"role": "key \| fill \| rim",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"azimuth\_deg": \-45.0,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"elevation\_deg": 30.0,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"clock\_direction": "4:30",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"intensity": 0.85,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"color\_rgb": \[1.0, 0.94, 0.88\],

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"kelvin": 5200,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"softness": 0.75,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"shape": "octagon \| circle\_ring \| rect\_softbox \| stripbox \| point",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"assigned\_execution": "physical\_lamp \| virtual\_shader"

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}

&nbsp;&nbsp;&nbsp;&nbsp;\]

&nbsp;&nbsp;},

&nbsp;&nbsp;"color\_dna": {

&nbsp;&nbsp;&nbsp;&nbsp;"lab\_mean": \[62.5, 4.2, 8.1\],

&nbsp;&nbsp;&nbsp;&nbsp;"lab\_std": \[18.4, 6.1, 9.5\],

&nbsp;&nbsp;&nbsp;&nbsp;"split\_toning": {

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"shadow\_tint\_rgb": \[0.08, 0.12, 0.18\],

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"highlight\_tint\_rgb": \[1.0, 0.95, 0.85\],

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"balance": 0.1

&nbsp;&nbsp;&nbsp;&nbsp;},

&nbsp;&nbsp;&nbsp;&nbsp;"luma\_quantiles": \[5, 22, 50, 78, 95\],

&nbsp;&nbsp;&nbsp;&nbsp;"fade\_amount": 0.06,

&nbsp;&nbsp;&nbsp;&nbsp;"grain\_amount": 0.12

&nbsp;&nbsp;}

}

## 4.2 五大并行 Agent 职责边界、输入输出与验收标准（Agent Work Packages）

&nbsp;

| Agent 模块 (/packages/\*) | 核心职责 | 技术栈与核心算子 | 输入契约 (Inputs) | 输出契约与独立验收标准 (Outputs & DoD) |
| :---- | :---- | :---- | :---- | :---- |
| Agent 1: ref-analyzer | 参考图逆向测光与色彩 DNA 提取器 | Python / ONNX \+ MediaPipe \+ DSINE \+ NNLS 非负最小二乘 \+ CIE Lab 统计 | 单张参考图像 Ref\_RGB (人像或商品) | 输出符合 Schema 的 TargetProfile.json；单图解析耗时 \<200ms，自带 3 张测试图单元校验。 |
| Agent 2: live-perception | 端侧实时几何感知、Greenroom 烘焙与信噪比护栏 | MediaPipe (FaceMesh \+ SelfieSeg) \+ 光流法线 EMA \+ 球谐离线烘焙器 | 实时相机流 Frame\_t \+ Greenroom 1.5s 转头序列 | 输出 u\_geo\_tex (RGBA16F: Normal+Depth)、u\_mask\_tex、64×64 u\_baked\_quotient\_matcap 及死黑预警事件。 |
| Agent 3: relight-shader | 60fps 混合 PBR 重打光渲染内核 | WebGL2 / OpenGL ES 3.0 Fragment Shaders (MatCap \+ 8-step SSRM \+ SSS \+ Catchlight) | TargetProfile.json \+ Agent 2 提供的纹理与烘焙贴图 | 独立 Storybook 画布跑满 1080p/4K @ 60fps (单帧 \<0.5ms\~8ms)，支持拖动虚拟光源实时预览鼻影与眼神光。 |
| Agent 4: studio-coach | 2D 表盘机位图生成、器材降维与 AR 差分教练 | React \+ 动态 SVG \+ Gemini 2.5 Flash Structured Output | TargetLight 与 CurrentLight 差值 ΔLight \+ 用户勾选现有灯具列表 | 渲染俯视时钟表盘 SVG 机位卡片；输出居家替代方案（如物理主灯+虚拟轮廓光分工）及 AR 挪灯箭头。 |
| Agent 5: grid-harmonizer | 朋友圈 9 图肤色锁仿色与九宫格美学自动排序器 | WebGL Lab 仿色 Shader \+ 棋盘格/C位放射/对角线组合排序求解器 | 9\~15 张用户原片 \+ TargetProfile.color\_dna \+ 皮肤 Masks | \<100ms 输出肤色自然通透的 9 张统一色调成片，并给出最佳 3×3 九宫格索引与废片剔除建议。 |

# 5. 端侧算子优化字典与性能护栏（On-Device Op Optimization & Guardrails）

为确保所有 Agent 编写的模型与后处理管线在移动端 GPU/NPU 上达到极致性能（Number of Delegated Partitions \= 1，零 CPU 回退），全体开发须严格遵循以下算子替换规范：

&nbsp;

| 算法/图形学意图 | 禁止使用的慢算子 / 反模式 | 强制替换的端侧黄金 Tensor/Shader Op | 性能收益与物理等价原因 |
| :---- | :---- | :---- | :---- |
| 掩码膨胀 / 腐蚀 / 环形边界提取 | 调用 OpenCV cv::dilate / findContours 触发 CPU-GPU 内存拷贝 | MaxPool2d(mask, k) / \-MaxPool2d(-mask, k) / MaxPool(m)-m | 100% 留在 GPU/NPU 计算图内，耗时 \<0.05ms，无动态 Shape。 |
| 硬边缘羽化 / 低频光场平滑扩散 | distanceTransform 或专用小型卷积网络 | 1/8 下采样 Interpolate \+ 循环 3\~8 次 AvgPool2d(3x3, replicate) | 邻域均值等价于离散拉普拉斯热传导方程，零参数权重，耗时 \~0.2ms。 |
| 高分辨率光照迁移与细节保留 | 将 1080p/4K 原图直接输入 U-Net / VAE 解码生成 RGB 像素 | 在 256×256 计算 Quotient Ratio Map，双线性上采样乘回 4K 原图 | 算力与带宽直降 64 倍，100% 保留原生 4K 毛孔、胡茬与织物纹理。 |
| 多光源漫反射与皮肤次表面散射 | 在 Fragment Shader 中逐像素循环计算多盏灯积分与高斯卷积 | 会前预烘焙为 64×64 球面查找表 (Quotient MatCap \+ Pre-Integrated SSS LUT) | 会中每像素仅需 1 次 texture2D(lut, N.xy\*0.5+0.5) 查表，耗时 \~0.1ms。 |
| 神经网络内部层结构设计 | Conv 7x7、Permute+Linear、LayerNorm、GELU、动态 MaskedSelect | Depthwise Conv3x3 \+ Conv1x1 \+ BatchNorm(折叠) \+ ReLU6/HardSwish \+ Where掩码 | MACs 降低 80%，支持 INT8 无损量化与硬件算子融合（Fusion）。 |

# 6. 分阶段演进里程碑与效果预期（Milestones & Expected Quality）

&nbsp;

| 阶段版本 | 开发周期与联调目标 | 核心交付特性 | 预期视觉效果评分 (对比专业 3 万元影棚实拍) |
| :---- | :---- | :---- | :---- |
| V0.1 (MVP 基础版) | 第 1 周（跑通 Agent 1 \+ Agent 4 \+ Agent 5） | 参考图逆向表盘机位图 SVG 生成 \+ 器材约束建议 \+ 朋友圈 9 图肤色锁仿色与九宫格自动排序。 | 75 分（9 图色调统一与机位拆解卡片已达商业发布与社交裂变水准）。 |
| V0.5 (实时 60fps 虚拟补光版) | 第 2\~3 周（接入 Agent 2 \+ Agent 3 商图像与 SSS） | 256×256 商图像 4K 保真乘回 \+ Pre-Integrated SSS 皮肤红润查表 \+ 虹膜 3D 眼神光动态追踪。 | 85 分（彻底消除塑料假脸感，普通室内平光下直出单反级通透人像视频流）。 |
| V1.0 (Greenroom 离线烘焙完全体) | 第 4 周（合入 Greenroom 预烘焙 \+ 2.5D 鼻影 \+ 通用物体 PBR） | 会前 1.5s 离线烘焙 Quotient MatCap（会中 \<0.5ms）+ 伦勃朗投射鼻影 \+ DSINE 任意商品材质重打光。 | 92\~96 分（结合拍前挪 1 盏物理主灯 \+ 端侧虚拟补齐辅/轮廓光，肉眼难辨真假）。 |

&nbsp;

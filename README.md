# Colab ComfyUI Tools

用 Colab 免费 T4 GPU 跑 ComfyUI + AnimateDiff 生成动画视频。

## 快速开始

### 1. 启动 Colab 服务器

点击下方链接，在 Colab 中打开 notebook：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FlashCDC/colab-comfyui-tools/blob/main/colab_comfyui_server.ipynb)

然后点击 **Runtime > Run all**，等待约 2-3 分钟。

看到 `Tunnel URL saved to Drive` 后，服务器就绑好了。

### 2. 本地生成视频

```bash
# 安装依赖
pip install requests

# 生成视频（Tunnel URL 从 Colab 复制）
python gen_video.py --url https://xxx.trycloudflare.com --image input.png

# 首次使用后 URL 会缓存到 ~/.comfyui_tunnel_url，下次不用再输
python gen_video.py --image input.png --prompt "cute cartoon, gentle motion"
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | 必填 | 输入图片路径 |
| `--prompt` | cute cartoon, warm colors... | 正向提示词 |
| `--negative` | ugly, blurry, low quality... | 负面提示词 |
| `--denoise` | 0.75 | 0.7=接近原图，0.85=更自由 |
| `--frames` | 16 | 16=约2秒，32=约4秒 |
| `--steps` | 20 | 采样步数，越多越精细 |
| `--cfg` | 7.0 | 引导强度 |
| `--width` | 512 | 宽度 |
| `--height` | 768 | 高度 |
| `--seed` | -1 | 随机种子，-1为随机 |
| `--output` | ./output | 输出目录 |

### 示例

```bash
# 基础生成
python gen_video.py --image photo.png

# 卡通风格，4秒视频
python gen_video.py --image cartoon.png --frames 32 --denoise 0.8

# 固定种子复现效果
python gen_video.py --image ref.png --seed 42 --denoise 0.7

# 竖屏 9:16
python gen_video.py --image portrait.png --width 448 --height 768 --frames 16
```

## 工作原理

```
┌─────────────────────────┐     ┌──────────────────────────┐
│  Colab (T4 GPU Server)  │     │  Local Mac (Client)     │
│                         │     │                          │
│  1. Mount Drive         │     │  1. gen_video.py         │
│  2. Install ComfyUI     │     │     --image input.png    │
│  3. Download models     │     │     --prompt "..."       │
│  4. Start server        │     │                          │
│  5. Cloudflare tunnel   │◄────│  2. Upload image         │
│     → tunnel_url.txt    │     │  3. Submit workflow      │
│  6. Wait for tasks      │     │  4. Poll status          │
│                         │────►│  5. Download video       │
└─────────────────────────┘     └──────────────────────────┘
```

## Colab 免费版限制

| 项目 | 限制 |
|------|------|
| GPU 时间 | 每天约 2-4 小时 |
| 单次 Session | 最长约 12 小时 |
| 空闲超时 | 约 90 分钟无操作断开 |
| 16帧生成 | 约 3-4 分钟 |
| 32帧生成 | 约 5-8 分钟 |

## 文件结构

```
colab-comfyui-tools/
├── colab_comfyui_server.ipynb   # Colab notebook (服务器)
├── gen_video.py                  # 本地客户端
└── README.md
```

## 常见问题

**Q: HuggingFace 下载超时？**
A: 在 Colab cell 里加代理或换镜像：
```python
!HF_ENDPOINT=https://hf-mirror.com wget ...
```

**Q: 400 Bad Request？**
A: 检查图片是否已上传（gen_video.py 会自动上传）。如果改了 workflow，检查节点类名和参数是否匹配当前 ComfyUI 版本。

**Q: Tunnel URL 过期？**
A: Colab 断开后 Tunnel 失效，需要重新跑 Colab notebook。

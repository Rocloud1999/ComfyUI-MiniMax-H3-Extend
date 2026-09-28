# Ref2VA + First/Last：不需要上一段视频

节点显示名：**MiniMax H3 Reference to Video + First/Last (Backported)**  
节点 ID：`MiniMaxH3ReferenceToVideoWithEndpointsPatched`

这是新增的独立节点；原来的 Video Extend / Encode AV 节点保留，不修改已有工作流的接口。

## 使用

在 ComfyUI 中搜索节点显示名，连接 H3 的 `clip`、视频 `vae`，填写 `prompt`、`width`、`height`、`length`。
`first_frame` 和 `last_frame` 至少连接一个；同时连接就是首尾帧引导。不需要 `context_latent`，也不需要把首图编码为假 context。

```text
H3 CLIP ─────────── clip
H3 Video VAE ────── vae
首帧 A ──────────── first_frame
尾帧 B ──────────── last_frame
参考图（可选）───── ref_images
                       │
       Reference to Video + First/Last
                       │
             positive / latent / frame_count
                       │
             现有 H3 Ref2VA 采样、解码流程
```

本节点不加载或替换 checkpoint；在现有 Ref2VA 工作流中替换条件构建节点即可，保留模型、采样器和音视频解码连接。
`width` / `height` 必须是 32 的倍数。默认 1344 × 768；不会因为输入图片较大而自动改变输出画布。

## 首尾、帧数与参考输入

首帧锚定目标帧 `0`；尾帧锚定**实际生成帧数减一**。`length` 经原生 `_empty_av_latent()` 对齐到 `17k+5`，例如请求 125 帧实际得到 141 帧，尾帧索引为 140。
`frame_count` 输出的是实际视频帧数，不是时间压缩后的 latent 长度。

遵循本仓库已有的端点预处理：首图拉伸到画布；尾图保持比例并居中覆盖裁剪。建议先把两张输入图准备为相同宽高比。
首尾端口收到 IMAGE batch 时仅使用第一张；多张参考图则把整个 IMAGE batch 接到 `ref_images`，每张成为一个 `<Picture i>`。
**首尾图不会自动加入参考图，也不会占用 `<Picture i>` 编号。**

可选 `ref_video` 接一个 24 fps 的 IMAGE 帧批次（至少 5 帧）；整个批次作为一个参考视频，而不是逐帧参考图，也不是要续写的前段视频。
`ref_video_audio` 是它的可选音轨，必须同时连接 `ref_video`。`ref_audio` 是独立参考音频。任一参考音频连接时都需要 H3 `audio_vae`；不使用参考音频时可以不连接它。
参考图片、视频和音频的编码及编号沿用 `nodes.py` 的 `_build_ref_blocks()`。

## 兼容机制

新节点复用原生 `_resize()`、`_empty_av_latent()` 和仓库的 reference helper。
首尾经 VAE 编码后进入 `minimax_keyframes`，参考内容进入 `minimax_refs`，二者保持分离。

沿用仓库现有能力检测：支持原生 keyframe layout 的 ComfyUI 不安装 legacy patch。
旧版给本节点的 keyframe 添加私有 `_h3_extend_endpoint` 标记，让 **PackedLayout 和 extra_conds 两处**兼容判断同时命中：

- 首尾坐标相对参考内容之后的 target origin 计算。
- 条件 latent 保持 `[first, last, references...]`，避免 refs 覆盖 keyframes。

该标记不是 context，不增加负时间帧；没有本插件标记的条件仍走原生路径。保留已有 native VideoExtend fork 的跳过保护。

## 验证与限制

已执行 Python 语法检查和 12 项 CPU 单元测试，包括端点位置、帧数取整、参考输入封装、旧版两处 dispatch、latent 顺序、时间原点、原生路径跳过、输入校验和注册入口。

```bash
python -m unittest discover -s tests -v
```

测试需要 PyTorch。ComfyUI、CLIP、VAE 和 reference helper 使用模拟对象；并非真实 H3 采样测试。
**尚未用真实 Ref2VA 权重、GPU 或视频输出验证视觉效果。**
这提供的是目标时间轴条件引导，不是采样后替换帧，也不保证首尾 RGB 逐像素一致。
生产使用前请用相同 seed 对比“仅 refs”与“refs + 首尾”的结果，并检查解码后的第 0 帧、最后一帧及相邻过渡。

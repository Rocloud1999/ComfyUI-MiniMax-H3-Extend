# Autogrow Ref2VA + First/Last

节点显示名：**MiniMax H3 Reference to Video + First/Last**  
节点 ID：`MiniMaxH3ReferenceToVideoWithEndpoints`

这是现有首尾帧节点的替换实现，不是第二个并行节点。使用官方同款 `io.ComfyNode`、`define_schema()`、`execute()` 和 Autogrow 输入组。原 Video Extend / Encode AV 节点不变。

## 输入与连接

连接 H3 `clip`、视频 `vae`，设置 `prompt`、`width`、`height`、`length`。`first_frame` 和 `last_frame` 至少连接一个；不需要上一段视频或 `context_latent`。

| 输入 | 用途 |
|---|---|
| `first_frame` | 目标第 0 帧锚点 |
| `last_frame` | 实际对齐后的最后一帧锚点 |
| `ref_images` | Autogrow，最多 9 个独立参考图片端口 |
| `ref_videos` | Autogrow，最多 3 个独立参考视频端口 |
| `ref_video_audios` | Autogrow，最多 3 个对应视频音轨端口 |
| `ref_audios` | Autogrow，最多 3 个独立参考音频端口 |
| `audio_vae` | 连接任意参考音频时必需 |

这些数量对齐本次核对的官方 `MiniMaxH3ReferenceToVideo` schema，是节点接口上限，不是模型架构上限。

每张参考图片直接连接一个动态端口，不再先合并成多图 batch。每个图片端口收到 batch 时，与官方一样只取第一张。独立图片可使用不同分辨率。

每段参考视频分别连接一个动态端口；该端口的 IMAGE batch 是这一段视频的完整帧序列，不是多段视频的集合。不要把多段视频拼接成同一个 batch。输入按 24 fps 准备，至少 5 帧；每段视频分别按生成长度上限及 `17k+5` 规则裁剪并编码。

视频音轨按端口的数字后缀配对。例如 `ref_video_audio_0` 对应 `ref_video_0`，不是按音轨连接顺序猜测。连接无对应视频的音轨会报错。动态端口后缀从 0 开始，但提示词中的 `<Picture 1>`、`<Video 1>`、`<Audio 1>` 标签仍按各类型实际参考素材从 1 编号。

参考顺序沿用官方及本仓库 helper：图片、视频（每个视频的音频标签在该视频标签前）、独立音频。跳过未连接端口，不重排收到的参考条目，也不修改输入字典。

## 首尾帧与输出

`first_frame` 锚定索引 0；`last_frame` 使用 `_empty_av_latent()` 返回的实际帧数减一。`length` 向上对齐到 `17k+5`，例如请求 125 帧得到 141 帧，尾帧索引是 140。

首图仍拉伸到指定画布；尾图仍保持比例并居中覆盖裁剪。两个端口都只取 batch 的第一张，并由原生 `_resize()` 转为 RGB。宽高仍要求 32 的倍数，默认 1344 × 768，不根据首图自动改变画布。

输出仅为 **`positive`、`latent`**，接入原有 H3 采样与音视频解码流程。`minimax_frame_count` 仍保留在 conditioning 元数据中供旧版模型布局使用；只是移除独立的 `frame_count` 输出端口，未删除推理所需的帧数信息。

首尾图只进入 `minimax_keyframes`，不会自动变成普通 reference，不占 `<Picture i>` 编号。本节点不加载、不替换 checkpoint，也不改变采样器。

## 推理逻辑与兼容边界

`nodes.py` 和 `patch.py` 未改动。新接口继续调用相同的 `_build_ref_blocks()`、`_empty_av_latent()`、`_resize()`，使用相同的 VAE 编码及 `clip.tokenize(..., minimax_ref_items=...)` 路径。

原生 keyframe layout 可用时，不安装 legacy patch。旧版仍给端点添加 `_h3_extend_endpoint` 标记，使现有两处补丁同时生效：目标时间原点包含所有参考跨度，视觉条件 latent 保持首尾在前、所有参考在后。补丁的作用范围、原生 VideoExtend fork 跳过保护和 continuation 机制都不变。

视频 VAE 仍必填，参考音频仍要求 audio VAE。这保留此前的真实 latent conditioning，不会因为缺少 VAE 而静默变成仅文本编码器参考；这一点比当前官方节点的可选 VAE 策略更严格。

需要支持 H3 和 V3 Autogrow 的 ComfyUI 及匹配的前端版本。没有 Autogrow API 时只跳过新首尾帧节点并给出警告，不提供旧首尾节点回退；原 Extend / Encode AV 的注册保留。前端太旧时也应更新，不能仅靠后端改代码获得动态端口。

注册使用同一个 `NODE_CLASS_MAPPINGS` 容器，里面同时放原 classic 节点和新的 V3 `ComfyNode`；新节点本身没有 classic `INPUT_TYPES` / `run()` 实现。ComfyUI 会优先处理这个注册表，因此没有另外放一个会被忽略的 `comfy_entrypoint`。

## 从此前的节点迁移

删除工作流中的旧首尾帧节点，重新搜索上面的新节点并连接。拆开原来送入 `ref_images` 的多图 batch，把每张图接入独立动态端口；原来的单个参考视频和音频接入相应输入组。输出重新连接 `positive`、`latent`。

旧节点 ID、单视频/单音频端口、批量图片拆分适配器和第三个输出端口都不再注册或保留。旧实现仅在 Git 历史中。此处的接口迁移不影响原 Video Extend / Encode AV 工作流。

## 验证

执行：

```bash
python -m py_compile __init__.py endpoint_nodes.py nodes.py patch.py tests/test_endpoints.py tests/check_history_parity.py
python -m unittest discover -s tests -v
python tests/check_history_parity.py
```

本次完成 21 项 CPU 测试，以及与历史提交 `eeb2d66004c9d52995c7561e4860a3651d5fe72a` 的 96 组等价输入比对，均通过。历史比对脚本从 Git 读取旧实现并检查 blob SHA，不在当前代码树中存放旧实现。浅克隆需要先取回该历史提交。

测试实际执行仓库的 reference helper、原续写节点和 legacy patch；V3 API、ComfyUI 几何/空 latent helper、CLIP、VAE、模型基础类使用测试替身。覆盖 9 张图 + 3 段视频 + 视频音轨 + 独立音频全部进入条件数据、各视频独立裁剪、音轨配对、首尾时间位置、latent 顺序、旧版补丁范围、原生路径不打补丁和原节点注册。

96 组比对检查等价输入下的 conditioning、初始 AV latent、tokenizer 参考条目及旧版 packed payload/layout；只改变接口，不改变这些推理准备结果。

**上述不是实际前端连线测试，也不是使用真实 H3 权重的 GPU 生成验证。** 未验证新多视频组合的最终视觉效果或性能。首尾是目标时间轴条件引导，不是输出帧替换，仍不保证 RGB 逐像素一致。

官方 schema 与注册行为核对基于 `Comfy-Org/ComfyUI@8cfe5e1ecb97512dea8deaac15e1228d7e6feeb1` 的 `nodes_minimax_h3.py`、`nodes.py` 和 `comfy_api/latest/_io.py`。

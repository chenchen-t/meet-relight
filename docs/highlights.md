# Meet-Relight: Key Highlights

## Executive Summary
"Studio lighting for your video calls: real-time, private, in the browser, powered by LiteRT.js and the Tensor API."

- **Main value:** studio-quality lighting for anyone, with no lights, no expertise and no server. It's private and runs in real time in a browser tab.
- **Why it's hard:** relighting has to finish inside one video frame, about 16 ms at 60 fps, alongside the ML models. Normally pre/post-processing runs as CPU glue code and the results are copied back from the GPU, which is slow and stalls the frame. Writing that glue as Tensor API graphs lets everything stay on the GPU and fit the budget.

---

## 🎯 Key Highlights

**In one line:** a virtual lighting studio in your browser.

### 1. The problem
- Most people don't know how to light themselves, and they don't own studio lights.
- Today's options are either too slow (AI image generation) or too heavy (needs a high-end laptop).

### 2. How it works
- **Small AI models see you:** MediaPipe finds your face, hair and body.
- **Graphics do the lighting:** your face shape comes from the face mesh, so no extra AI model is needed.
- **1.5 s Greenroom setup:** it learns your room's light, so it can remove harsh shadows as well as add new light.
- **Only the light changes:** skin, texture and makeup are kept exactly.
- **No lag:** about 8 ms added, and the light always stays on your face.

### 3. ⭐ Tensor API
- Pre/post-processing is 5 tiny graphs (< 70 KB) written with the Tensor API.
- They run together with the MediaPipe models on LiteRT.js + WebGPU.
- Everything stays on the GPU: nothing is copied back to the CPU.
- Write once, run on any platform.

### 4. ⭐ Gemma 4 on-device
- Say "Wong Kar-wai neon" or "less shadow", or show a reference photo, and Gemma 4 sets the lights.
- It runs in the browser without freezing the video.

### 5. Private
- 100% on-device: no video leaves your computer.

> *"MediaPipe models + LiteRT runtime + Tensor API glue = one GPU pipeline from camera to pixels. Add on-device Gemma 4, and one sentence sets up your lighting."*

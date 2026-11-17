# 🚀 Ethiopian Birr AI - Web Deployment Guide

This folder contains a **100% static, client-side deployable website**.
The YOLOv8 AI model runs **directly inside the user's browser** via WebAssembly (WASM) and WebGL using **ONNX Runtime Web**.

No Python backend, no Docker, and no server maintenance are required!

---

## ⚡ Option 1: Instant 30-Second Free Deploy (Netlify Drop)
1. Go to **[Netlify Drop](https://app.netlify.com/drop)** in your browser.
2. Drag and drop the **`web_app`** folder directly into the browser window.
3. Done! Netlify immediately gives you a live public URL (e.g., `https://ethiopian-birr-ai.netlify.app`) that works on any phone, tablet, or laptop.

---

## ⚡ Option 2: Deploy to Vercel
1. Install Vercel CLI (optional) or push this repository to GitHub.
2. Go to **[Vercel Dashboard](https://vercel.com/new)** and import your repository.
3. Set the **Root Directory** to `web_app`.
4. Click **Deploy**. Vercel will build and serve your static site globally with SSL.

---

## ⚡ Option 3: Deploy to GitHub Pages (100% Free)
1. In your GitHub repository settings, go to **Settings** -> **Pages**.
2. Under **Build and deployment** -> **Source**, select `Deploy from a branch`.
3. Select your branch and set the folder to `/web_app` (or move the contents of `web_app` to the root).
4. Save, and your website will be live at `https://<your-username>.github.io/<repo-name>/`.

---

## 💻 Local Testing (Optional)
Because browsers restrict fetching binary model weights (`.onnx`) over `file:///` URLs due to CORS security policies, serve the folder with any static local server:

- **Using Python**:
  ```bash
  cd web_app
  python -m http.server 8000
  ```
  Then open `http://localhost:8000` in your browser.

- **Using Node.js**:
  ```bash
  npx serve web_app
  ```

- **Or simply open `index.html`**:
  If opened directly via `file:///`, click **"Choose 'best.onnx'"** on the startup screen to load the model file directly from your disk!

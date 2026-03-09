# Frontend Design & Architecture Document
## Project: TwinVision (Prompt-to-Video Model Comparison)
**Target AI Developer:** Antigravity Gemini 3.1 Pro
**Reference Design:** MoncyDev Portfolio (moncy.dev) - specifically targeting its dark mode, typography-heavy, scroll-animated, 3D/WebGL-inspired creative developer aesthetic.

---

## 1. Design System & Theme

### Colors
- **Background:** Deep rich black (`#050505` to `#0a0a0a`)
- **Primary Accents:** Neon Green/Cyberpunk Lime (`#ccff00` or `#39ff14`) and Electric Blue (`#00f0ff`) for highlights.
- **Text:** 
  - Primary: Off-white/Cream (`#f0f0f0`)
  - Secondary: Muted Gray (`#888888`)
- **Surface/Cards:** Glassmorphism with very low opacity white (`rgba(255, 255, 255, 0.03)`), 1px solid border (`rgba(255, 255, 255, 0.08)`), heavy backdrop-blur.

### Typography (Like moncy.dev)
- **Headings:** Massive, bold, uppercase sans-serif (e.g., *Monument Extended*, *Clash Display*, or *Inter Tight*). 
- **Body:** Clean, legible sans-serif (e.g., *Inter* or *Manrope*).
- **Style:** Use outline text effects (web-kit-text-stroke) and marquee scrolling text for sections.

### Animations & Vibe (GSAP / Framer Motion)
- **Preloader:** A full-screen percentage counter counting 0 to 100% with a staggered reveal of the hero section.
- **Scroll Effects:** Elements fade and slide up on scroll. Text reveals character-by-character or line-by-line.
- **Hover States:** Magnetic buttons, image distortion on hover, custom cursor (a small dot that expands when hovering over clickable items).

---

## 2. Tech Stack for Frontend
- **Framework:** React + TypeScript (Vite or Next.js)
- **Styling:** Tailwind CSS + standard CSS for specific text-stroke/marquee effects.
- **Animations:** GSAP (ScrollTrigger) or Framer Motion.
- **3D/WebGL (Optional but recommended for the vibe):** React Three Fiber / Drei (for a subtle interactive particle or fluid background).

---

## 3. Page Structure & Components

### A. Preloader (Initial Load)
- Big bold text: `LOADING [XX]%`
- Marquee text behind it: `TWINVISION // AI MODEL COMPARISON`

### B. Header / Navigation
- Fixed, glassmorphic navbar.
- Left: Logo `TWINVISION.`
- Center: Links (`[ THE PIPELINE ]`, `[ COMPARISON ]`, `[ METRICS ]`) with a strike-through hover effect.
- Right: Glowing GitHub link.

### C. Hero Section
- **Headline:** Huge typography taking up the screen. 
  - Line 1: `PROMPT`
  - Line 2: `TO VIDEO` (Outline text, hollow)
  - Line 3: `EVALUATION.`
- **Interactive Element:** A sleek "terminal-like" input box where the user can type a prompt (e.g., "AI robots exploring Mars"). When they click "Generate Pipeline", it triggers a loading animation that simulates the backend processing.

### D. The Pipeline (Horizontal Scroll Section)
- Use GSAP ScrollTrigger to create a horizontal scrolling section.
- 4 massive cards:
  1. `01 / PROMPT INPUT`
  2. `02 / IMAGE GEN (FLUX VS SD3.5)`
  3. `03 / FFMPEG COMPILATION`
  4. `04 / AUTOMATED METRICS`
- Each card has a subtle wireframe/grid background.

### E. The Comparison Showcase (Main UI)
- **Layout:** Split screen (50/50).
- **Left Side:** FLUX.1 [schnell] output.
  - Video player or image carousel.
  - Small neon tag: `WINNER: CLARITY`
- **Right Side:** Stable Diffusion 3.5 output.
  - Video player or image carousel.
  - Small neon tag: `WINNER: ART STYLE`
- **Center:** An interactive before/after slider handle (if comparing specific frames), or just clean spacing.

### F. Analytics & Metrics Dashboard
- A "Bento Box" grid layout.
- **Card 1 (Wide):** Bar chart comparing CLIP Scores (using Recharts or Chart.js styled in dark mode).
- **Card 2 (Square):** Big number `BRISQUE` score with a red/green indicator.
- **Card 3 (Square):** Big number `NIQE` score.
- **Card 4 (Wide):** LPIPS & SSIM temporal consistency graphs.
- **Style:** Make the data look like a futuristic control panel.

### G. Footer
- Massive scrolling marquee: `FLUX.1 • STABLE DIFFUSION • FFMPEG • CLAUDE CODE •`
- Simple links to the developer (you) and college project details.

---

## 4. Instructions for Gemini 3.1 Pro

**Prompt for Gemini:**
> "I am building a React/TypeScript frontend for an AI project called TwinVision. I want you to act as an expert Creative Developer. Read the design document provided above. 
> 
> Your task is to write the complete frontend code (using Vite, React, Tailwind, and Framer Motion/GSAP). Start by giving me the `App.tsx` structure and the main Hero and Comparison components. 
> 
> Crucially, implement the 'moncy.dev' aesthetic: dark mode, huge typography with outline effects, glassmorphic cards, and smooth scroll animations. Ensure there are placeholder functions (e.g., `handleGenerate()`) where my Claude Code backend will hook into the Python pipeline."

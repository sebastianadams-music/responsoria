import { ResponsoriaAudioEngine } from "./audioEngine.js";

const canvas = document.getElementById("stage");
const ctx = canvas.getContext("2d");
const promptEl = document.getElementById("prompt");

let width = window.innerWidth;
let height = window.innerHeight;
let dpr = Math.min(window.devicePixelRatio || 1, 2);

const userPos = { x: 0.5, y: 0.5 };
const compPos = { x: 0.5, y: 0.5 };
const compTarget = { x: 0.5, y: 0.5 };

let currentMode = "user";
let bgLightness = 0; // 0 = black (user mode), 1 = white (answer mode)
let bgTarget = 0;
let lastFrameTime = performance.now();

const ripples = [];

function resize() {
  width = window.innerWidth;
  height = window.innerHeight;
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

resize();
window.addEventListener("resize", resize);

const engine = new ResponsoriaAudioEngine({
  onModeChange(mode) {
    currentMode = mode;
    bgTarget = mode === "answer" ? 1 : 0;
    document.body.classList.toggle("mode-user", mode === "user");
    document.body.classList.toggle("mode-answer", mode === "answer");
  },
  onNoteTriggered({ mode, x, y, amplitude }) {
    if (mode === "answer") {
      compTarget.x = x;
      compTarget.y = y;
      ripples.push({
        x,
        y,
        mode,
        born: performance.now(),
        life: 440,
        amp: amplitude,
      });
    } else {
      ripples.push({
        x: userPos.x,
        y: userPos.y,
        mode,
        born: performance.now(),
        life: 400,
        amp: amplitude,
      });
    }
  },
});

function updatePointer(clientX, clientY) {
  const x = Math.min(1, Math.max(0, clientX / Math.max(1, width)));
  const y = Math.min(1, Math.max(0, 1 - clientY / Math.max(1, height)));
  userPos.x = x;
  userPos.y = y;
  engine.setUserPosition(x, y);
}

let startedOnce = false;

function tryUnlockAudio() {
  engine.unlockFromGesture();
  if (!startedOnce) {
    startedOnce = true;
    if (promptEl) promptEl.classList.add("hidden");
  }
  engine.start();
}

window.addEventListener("pointermove", (e) => {
  updatePointer(e.clientX, e.clientY);
});

window.addEventListener("pointerdown", (e) => {
  updatePointer(e.clientX, e.clientY);
  tryUnlockAudio();
});

window.addEventListener(
  "touchstart",
  (e) => {
    if (e.touches && e.touches[0]) {
      updatePointer(e.touches[0].clientX, e.touches[0].clientY);
    }
    tryUnlockAudio();
  },
  { passive: true }
);

window.addEventListener(
  "touchend",
  () => {
    tryUnlockAudio();
  },
  { passive: true }
);

window.addEventListener("click", () => {
  tryUnlockAudio();
});

window.addEventListener("keydown", () => {
  tryUnlockAudio();
});

// Preload metadata immediately; wait for user interaction to start audio
engine.preloadMetadata();

function render(now) {
  const dt = Math.min(0.1, (now - lastFrameTime) / 1000);
  lastFrameTime = now;

  // Smooth ~350ms background interpolation
  const bgRate = 1 - Math.exp(-dt * 9.5);
  bgLightness += (bgTarget - bgLightness) * bgRate;

  // Smooth glide for computer point in answer mode
  const glideRate = 1 - Math.exp(-dt * 12);
  compPos.x += (compTarget.x - compPos.x) * glideRate;
  compPos.y += (compTarget.y - compPos.y) * glideRate;

  const bgVal = Math.round(bgLightness * 255);
  ctx.fillStyle = `rgb(${bgVal}, ${bgVal}, ${bgVal})`;
  ctx.fillRect(0, 0, width, height);

  // Subtle fleeting ripples on triggered notes
  for (let i = ripples.length - 1; i >= 0; i--) {
    const r = ripples[i];
    const age = now - r.born;
    if (age >= r.life) {
      ripples.splice(i, 1);
      continue;
    }
    const p = age / r.life;
    const radius = 5 + p * (18 + r.amp * 14);
    const alpha = (1 - p) * (1 - p) * 0.32;
    const px = r.x * width;
    const py = (1 - r.y) * height;
    const strokeVal = r.mode === "answer" ? 0 : 255;

    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(${strokeVal}, ${strokeVal}, ${strokeVal}, ${alpha.toFixed(3)})`;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Active point: white in user mode at cursor/touch, black in answer mode at selected timbre
  const active = currentMode === "user" ? userPos : compPos;
  const ptX = active.x * width;
  const ptY = (1 - active.y) * height;
  const ptColor = currentMode === "user" ? "#ffffff" : "#000000";

  ctx.beginPath();
  ctx.arc(ptX, ptY, 5, 0, Math.PI * 2);
  ctx.fillStyle = ptColor;
  ctx.fill();

  requestAnimationFrame(render);
}

requestAnimationFrame(render);

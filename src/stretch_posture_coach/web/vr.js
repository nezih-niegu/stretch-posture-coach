// Stretch telepresence for the Meta Quest browser (WebXR + three.js).
//
// Every RGB-D frame from the robot's head camera becomes a textured 3-D surface (depth displaces a grid in the
// vertex shader). WebXR renders it for each eye, so stereo and head-motion parallax are real and run at the
// headset's refresh rate. Surfaces are placed with the robot-head yaw/pitch at capture time (world-locked).
// Your head direction is sent back so the robot head looks where you look.

import * as THREE from 'three';

const $ = (id) => document.getElementById(id);
const msg = (s) => { $('msg').textContent = s; };

// ------------------------------------------------------------------ renderer / scene
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local');
if (renderer.xr.setFoveation) renderer.xr.setFoveation(0.4);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0c10);
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.05, 80);
scene.add(camera);

const world = new THREE.Group();   // robot frame: origin = robot head camera, -Z = robot forward
scene.add(world);
const ui = new THREE.Group();      // panels in front of the user
scene.add(ui);

const floor = new THREE.GridHelper(20, 40, 0x3a3f4a, 0x22252c);
floor.position.y = -1.3;
floor.material.transparent = true;
floor.material.opacity = 0.5;
world.add(floor);

// ------------------------------------------------------------------ RGB-D surface
const VS = /* glsl */`
  uniform sampler2D depthTex;
  uniform vec4 intr;          // fx/W, fy/H, cx/W, cy/H
  uniform vec2 texel;
  uniform float maxRange;
  uniform float jump;
  varying vec2 vUv;
  varying float vValid;
  void main() {
    vec2 st = uv;
    float z  = texture2D(depthTex, st).r;
    float zr = texture2D(depthTex, st + vec2(texel.x, 0.0)).r;
    float zd = texture2D(depthTex, st + vec2(0.0, texel.y)).r;
    float zl = texture2D(depthTex, st - vec2(texel.x, 0.0)).r;
    float zu = texture2D(depthTex, st - vec2(0.0, texel.y)).r;
    float m = max(max(abs(z - zr), abs(z - zd)), max(abs(z - zl), abs(z - zu)));
    vValid = (z > 0.15 && z < maxRange && m < jump * z) ? 1.0 : 0.0;
    vec3 p = vec3((st.x - intr.z) / intr.x * z, -(st.y - intr.w) / intr.y * z, -z);
    vUv = st;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }`;
const FS = /* glsl */`
  uniform sampler2D colorTex;
  uniform float brightness;
  uniform float opacity;
  varying vec2 vUv;
  varying float vValid;
  void main() {
    if (vValid < 0.99) discard;
    vec3 c = texture2D(colorTex, vUv).rgb * brightness;
    gl_FragColor = vec4(c, opacity);
    #include <colorspace_fragment>
  }`;

function gridGeometry(dw, dh) {
  const pos = new Float32Array(dw * dh * 3);  // real positions come from the shader
  const uv = new Float32Array(dw * dh * 2);
  for (let j = 0; j < dh; j++) for (let i = 0; i < dw; i++) {
    const k = j * dw + i;
    uv[2 * k] = (i + 0.5) / dw;
    uv[2 * k + 1] = (j + 0.5) / dh;           // v grows downward (textures uploaded top row first)
  }
  const idx = [];
  for (let j = 0; j < dh - 1; j++) for (let i = 0; i < dw - 1; i++) {
    const a = j * dw + i, b = a + 1, c = a + dw, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 100);  // shader moves vertices; never cull
  return g;
}

function poseMatrix(yaw, pitch) {
  const m = new THREE.Matrix4().makeRotationY(yaw);
  return m.multiply(new THREE.Matrix4().makeRotationX(pitch));
}

class Surface {
  constructor(dw, dh, brightness) {
    this.dw = dw; this.dh = dh;
    this.depth = new THREE.DataTexture(new Float32Array(dw * dh), dw, dh, THREE.RedFormat, THREE.FloatType);
    this.depth.minFilter = this.depth.magFilter = THREE.NearestFilter;
    this.color = new THREE.Texture();
    this.color.flipY = false;
    this.color.colorSpace = THREE.SRGBColorSpace;
    this.color.minFilter = THREE.LinearFilter;
    this.color.generateMipmaps = false;
    this.material = new THREE.ShaderMaterial({
      vertexShader: VS, fragmentShader: FS, transparent: brightness < 1,
      uniforms: {
        depthTex: { value: this.depth }, colorTex: { value: this.color }, intr: { value: new THREE.Vector4() },
        texel: { value: new THREE.Vector2(1 / dw, 1 / dh) }, maxRange: { value: 6 }, jump: { value: 0.06 },
        brightness: { value: brightness }, opacity: { value: 1 },
      },
    });
    this.mesh = new THREE.Mesh(gridGeometry(dw, dh), this.material);
    this.mesh.matrixAutoUpdate = false;
    this.mesh.frustumCulled = false;
    this.bitmap = null;
    this.person = new Uint8Array(dw * dh);  // 1 = person pixel (never kept in the scene memory)
    this.yaw = 0; this.pitch = 0; this.time = 0;
  }
  update(meta, depthMM, bitmap) {
    const d = this.depth.image.data;
    for (let i = 0; i < d.length; i++) {
      const v = depthMM[i];
      this.person[i] = v & 0x8000 ? 1 : 0;
      d[i] = (v & 0x7fff) * 0.001;
    }
    this.depth.needsUpdate = true;
    if (this.bitmap && this.bitmap !== bitmap && !this.keepBitmap) this.bitmap.close();
    this.bitmap = bitmap;
    this.color.image = bitmap;
    this.color.needsUpdate = true;
    const u = this.material.uniforms;
    u.intr.value.set(...meta.intr);
    u.maxRange.value = meta.max_range || 6;
    this.yaw = meta.yaw; this.pitch = meta.pitch; this.time = performance.now();
    this.mesh.matrix.copy(poseMatrix(meta.yaw, meta.pitch));
    this.mesh.matrixWorldNeedsUpdate = true;
  }
  cloneAsMemory() {
    const s = new Surface(this.dw, this.dh, 0.6);
    const src = this.depth.image.data, dst = s.depth.image.data;
    for (let i = 0; i < dst.length; i++) dst[i] = this.person[i] ? 0 : src[i];  // people are not "scenery"
    s.depth.needsUpdate = true;
    s.color.image = this.bitmap; s.color.needsUpdate = true; s.bitmap = this.bitmap;
    this.keepBitmap = false; s.keepBitmap = true;
    s.material.uniforms.intr.value.copy(this.material.uniforms.intr.value);
    s.material.uniforms.maxRange.value = this.material.uniforms.maxRange.value;
    s.material.polygonOffset = true; s.material.polygonOffsetFactor = 2; s.material.polygonOffsetUnits = 2;
    s.yaw = this.yaw; s.pitch = this.pitch; s.time = performance.now();
    s.mesh.matrix.copy(this.mesh.matrix);
    s.mesh.renderOrder = -1;
    return s;
  }
  dispose() {
    this.mesh.geometry.dispose(); this.material.dispose(); this.depth.dispose(); this.color.dispose();
    if (this.bitmap && this.keepBitmap) this.bitmap.close();
  }
}

let live = null;
const memory = [];          // older views kept around you (dimmed)
let memoryOn = true;
const MAX_MEMORY = 8;
const MEMORY_MAX_AGE_MS = 120000;  // older views fade out and are dropped
let stableFrames = 0, lastPose = null;
const angDist = (a, b) => Math.hypot(Math.atan2(Math.sin(a.yaw - b.yaw), Math.cos(a.yaw - b.yaw)), a.pitch - b.pitch);

function rememberLive() {
  if (!live || !live.bitmap) return;
  if (stableFrames < 3) return;  // only remember sharp views: the head has been still for a few frames
  const near = memory.filter((m) => angDist(m, live) < THREE.MathUtils.degToRad(15));
  const fresh = near.some((m) => performance.now() - m.time < 20000);
  if (fresh) return;
  near.forEach((m) => { world.remove(m.mesh); m.dispose(); memory.splice(memory.indexOf(m), 1); });
  live.keepBitmap = true;          // the memory now owns this bitmap
  const m = live.cloneAsMemory();
  live.bitmap = null;
  memory.push(m);
  world.add(m.mesh);
  while (memory.length > MAX_MEMORY) { const old = memory.shift(); world.remove(old.mesh); old.dispose(); }
}

function updateMemoryVisibility() {
  for (const m of memory) {
    // hide memories the live view is covering (avoids double images); everything else stays as context
    const age = performance.now() - m.time;
    m.material.uniforms.brightness.value = 0.6 * Math.max(0.25, 1 - age / MEMORY_MAX_AGE_MS);
    m.mesh.visible = memoryOn && (!live || angDist(m, live) > THREE.MathUtils.degToRad(35));
  }
  for (let i = memory.length - 1; i >= 0; i--) {
    if (performance.now() - memory[i].time > MEMORY_MAX_AGE_MS) {
      world.remove(memory[i].mesh); memory[i].dispose(); memory.splice(i, 1);
    }
  }
}

// ------------------------------------------------------------------ skeleton (3-D, colored by risk)
const LEVEL_COLORS = [new THREE.Color(0x50c850), new THREE.Color(0xffc800), new THREE.Color(0xeb3c3c)];
const NEUTRAL = new THREE.Color(0x3cb4ff);
const skelGeo = new THREE.BufferGeometry();
skelGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(64 * 3), 3));
skelGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(64 * 3), 3));
const skeleton = new THREE.LineSegments(skelGeo, new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false }));
skeleton.matrixAutoUpdate = false;
skeleton.renderOrder = 10;
skeleton.frustumCulled = false;
world.add(skeleton);
const REGION = (a, b) => {
  const s = new Set([a, b]);
  if ([0, 7, 8].some((i) => s.has(i))) return 'neck';
  if ([13, 15].some((i) => s.has(i))) return 'l_arm';
  if ([14, 16].some((i) => s.has(i))) return 'r_arm';
  return 'trunk';
};
function updateSkeleton(meta) {
  const pts = meta.skeleton;
  if (!pts) { skeleton.visible = false; return; }
  const pos = skelGeo.attributes.position.array, col = skelGeo.attributes.color.array;
  let n = 0;
  for (const [a, b] of meta.edges) {
    const pa = pts[a], pb = pts[b];
    if (!pa || !pb) continue;
    const lvl = meta.levels ? meta.levels[REGION(a, b)] : -1;
    const c = lvl >= 0 ? LEVEL_COLORS[lvl] : NEUTRAL;
    for (const p of [pa, pb]) {
      pos[3 * n] = p[0]; pos[3 * n + 1] = -p[1]; pos[3 * n + 2] = -p[2];
      col[3 * n] = c.r; col[3 * n + 1] = c.g; col[3 * n + 2] = c.b;
      n++;
    }
  }
  skelGeo.setDrawRange(0, n);
  skelGeo.attributes.position.needsUpdate = true;
  skelGeo.attributes.color.needsUpdate = true;
  skeleton.matrix.copy(poseMatrix(meta.yaw, meta.pitch));
  skeleton.matrixWorldNeedsUpdate = true;
  skeleton.visible = n > 0;
}

// ------------------------------------------------------------------ panels: HUD, gripper cam, nav backdrop
function canvasPanel(wPx, hPx, wM) {
  const canvas = document.createElement('canvas');
  canvas.width = wPx; canvas.height = hPx;
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(wM, wM * hPx / wPx),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthTest: false }));
  mesh.renderOrder = 20;
  return { canvas, ctx: canvas.getContext('2d'), tex, mesh };
}
const hud = canvasPanel(1024, 680, 0.62);
hud.mesh.position.set(0.62, -0.38, -1.05);
hud.mesh.lookAt(0, 0, 0);
ui.add(hud.mesh);

const gripTex = new THREE.Texture();
gripTex.colorSpace = THREE.SRGBColorSpace; gripTex.flipY = false; gripTex.generateMipmaps = false;
const gripper = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 0.28),
  new THREE.MeshBasicMaterial({ map: gripTex, depthTest: false }));
gripper.position.set(-0.62, -0.38, -1.05);
gripper.lookAt(0, 0, 0);
gripper.renderOrder = 20;
gripper.visible = false;
ui.add(gripper);
// three's PlaneGeometry has v up; our bitmaps are top-row-first with flipY=false -> flip the uvs once
for (const g of [gripper.geometry]) {
  const uv = g.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setY(i, 1 - uv.getY(i));
}

const navTex = new THREE.Texture();
navTex.colorSpace = THREE.SRGBColorSpace; navTex.flipY = false; navTex.generateMipmaps = false;
let nav = null;
function ensureNav(meta) {
  const hfov = THREE.MathUtils.degToRad(meta.hfov || 120), vfov = THREE.MathUtils.degToRad(meta.vfov || 90);
  if (nav && nav.userData.key === `${meta.hfov}/${meta.vfov}`) return;
  if (nav) { world.remove(nav); nav.geometry.dispose(); }
  const r = 12;
  const geo = new THREE.CylinderGeometry(r, r, 2 * r * Math.tan(vfov / 2), 48, 1, true, Math.PI - hfov / 2, hfov);
  const uv = geo.attributes.uv;
  for (let i = 0; i < uv.count; i++) { uv.setX(i, 1 - uv.getX(i)); uv.setY(i, 1 - uv.getY(i)); }
  nav = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: navTex, side: THREE.BackSide, depthWrite: false,
    color: 0x6e6e6e }));  // dim: peripheral context only, the 3-D scene stays in front
  nav.renderOrder = -5;
  nav.matrixAutoUpdate = false;
  nav.userData.key = `${meta.hfov}/${meta.vfov}`;
  world.add(nav);
}

let state = { headControl: true, lastHud: {}, fps: 0, latency: 0, connected: false, alarm: { active: false } };
function wrap(ctx, text, x, y, maxW, lh, maxLines) {
  const words = String(text).split(' ');
  let line = '', n = 0;
  for (const w of words) {
    const t = line ? `${line} ${w}` : w;
    if (ctx.measureText(t).width > maxW && line) {
      ctx.fillText(line, x, y); y += lh; line = w; if (++n >= maxLines) return y;
    } else line = t;
  }
  if (line) { ctx.fillText(line, x, y); y += lh; }
  return y;
}
function drawHud() {
  const { ctx, canvas } = hud;
  const h = state.lastHud || {};
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'rgba(16,17,22,0.86)';
  ctx.beginPath(); ctx.roundRect(0, 0, canvas.width, canvas.height, 28); ctx.fill();
  ctx.fillStyle = '#ffd27a'; ctx.font = 'bold 40px system-ui'; ctx.fillText('STRETCH POSTURE COACH', 32, 58);
  ctx.font = '28px system-ui'; ctx.fillStyle = '#9aa0aa';
  ctx.fillText(`${state.connected ? 'live' : 'offline'} · ${state.fps.toFixed(0)} fps · ${state.latency.toFixed(0)} ms`, 32, 98);
  let y = 150;
  const risk = (s) => (s == null ? '#ccc' : s >= 5 ? '#eb3c3c' : s >= 3 ? '#ffc800' : '#50c850');
  if (h.present === false) {
    ctx.fillStyle = '#ffc800'; ctx.font = '34px system-ui'; ctx.fillText('No person in view', 32, y); y += 50;
  } else if (h.present) {
    ctx.font = '34px system-ui'; ctx.fillStyle = '#eee'; ctx.fillText(`Posture: ${h.posture || '?'}`, 32, y);
    if (h.rula) { ctx.fillStyle = risk(h.rula.score); ctx.fillText(`RULA ${h.rula.score} (${h.rula.risk})`, 400, y); }
    y += 48;
    if (h.reba) { ctx.fillStyle = risk(h.reba.score - 2); ctx.fillText(`REBA ${h.reba.score} (${h.reba.risk})`, 400, y); }
    ctx.fillStyle = '#eee';
    ctx.fillText(`Neck ${h.neck_flexion ?? '-'}°  Trunk ${h.trunk_flexion ?? '-'}°`, 32, y); y += 48;
    ctx.fillText(`L5/S1 ${h.l5s1_compression_n ?? '-'} N`, 32, y); y += 52;
    ctx.fillStyle = '#eb6b6b';
    for (const issue of (h.issues || []).slice(0, 3)) { ctx.fillText(`• ${issue}`, 32, y); y += 42; }
  }
  const f = h.follow;
  if (f && f.distance_m != null) {
    ctx.fillStyle = f.distance_m >= 2 && f.distance_m <= 3 ? '#50c850' : '#ffc800';
    ctx.font = '30px system-ui';
    ctx.fillText(`Distance ${f.distance_m.toFixed(2)} m${f.enabled ? ` · follow: ${f.action}` : ''}`, 32, y); y += 46;
  }
  if (h.last_message) {
    ctx.fillStyle = '#ffd27a'; ctx.font = 'italic 28px system-ui';
    y = wrap(ctx, `“${h.last_message}”`, 32, y + 6, canvas.width - 64, 36, 3);
  }
  ctx.fillStyle = '#9aa0aa'; ctx.font = '24px system-ui';
  ctx.fillText(`A/X head follows you: ${state.headControl ? 'ON' : 'off'} · B/Y memory: ${memoryOn ? 'ON' : 'off'} · trigger: recenter`,
    32, canvas.height - 24);
  const al = state.alarm;
  if (al.active) {  // flashing fall banner on top of everything
    const on = Math.floor(performance.now() / 500) % 2 === 0 || al.state !== 'alarm';
    ctx.fillStyle = al.state === 'alarm' ? (on ? '#d03b3b' : '#7a1010') : '#c77700';
    ctx.fillRect(0, 0, canvas.width, 150);
    ctx.fillStyle = '#fff'; ctx.font = 'bold 46px system-ui';
    ctx.fillText(al.state === 'alarm' ? 'FALL ALARM' : 'Possible fall - checking', 32, 60);
    ctx.font = '26px system-ui';
    wrap(ctx, `${al.reason || ''} · squeeze grip = I'm OK`, 32, 104, canvas.width - 64, 32, 2);
  }
  hud.tex.needsUpdate = true;
}
setInterval(drawHud, 250);

// ------------------------------------------------------------------ networking
let ws = null;
let frames = 0, lastFpsT = performance.now();
const decoder = new TextDecoder();

function connect() {
  const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { state.connected = true; msg('connected - waiting for video…'); };
  ws.onclose = () => { state.connected = false; msg('disconnected - retrying…'); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') return onState(JSON.parse(ev.data));
    onBinary(ev.data);
  };
}

function ackAlarm() {
  if (ws && ws.readyState === 1 && state.alarm.active) ws.send(JSON.stringify({ type: 'alarm_ack' }));
}
$('ack').onclick = ackAlarm;

function onAlarm(a) {
  state.alarm = a;
  const box = $('alarm');
  box.style.display = a.active ? 'block' : 'none';
  box.className = a.state === 'alarm' ? '' : 'check';
  $('atitle').textContent = a.state === 'alarm' ? 'FALL ALARM' : 'Possible fall - checking…';
  $('areason').textContent = a.reason || '';
  if (a.active) {
    hud.mesh.position.set(0, -0.15, -0.9);   // bring the HUD to the middle of the view while alarming
    hud.mesh.lookAt(0, 0, 0);
  } else {
    hud.mesh.position.set(0.62, -0.38, -1.05);
    hud.mesh.lookAt(0, 0, 0);
  }
}

$('endsess').onclick = () => {
  if (!confirm('End this session and save its report? Stretch keeps running.')) return;
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'end_session' }));
};
function onReport(r) {
  $('saved').innerHTML = r.status === 'ready'
    ? `Saved ${r.name} · <a href="${r.url}" target="_blank">report</a>` : `Saving ${r.name}…`;
}

function onState(s) {
  if (s.type === 'alarm') return onAlarm(s);
  if (s.type === 'report') return onReport(s);
  if (s.type === 'state' && s.state === 'ended') {
    msg('Session ended - opening the report…');
    const go = () => { location.href = s.report || '/sessions'; };
    const session = renderer.xr.getSession();
    if (session) session.end().then(go, go); else go();
  }
}

function ack(type) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'ack', t: type }));
}

async function onBinary(buf) {
  const type = new DataView(buf).getUint8(0);
  try {
    await handleBinary(buf, type);
  } finally {
    ack(type);   // flow control: the robot only sends the next frame once we've shown this one
  }
}

async function handleBinary(buf, type) {
  const dv = new DataView(buf);
  const jl = dv.getUint32(1, true);
  const meta = JSON.parse(decoder.decode(new Uint8Array(buf, 5, jl)));
  let off = 5 + jl;
  if (type === 1) {
    const il = dv.getUint32(off, true); off += 4;
    const jpg = new Blob([new Uint8Array(buf, off, il)], { type: 'image/jpeg' }); off += il;
    const depth = new Uint16Array(buf.slice(off));
    const bitmap = await createImageBitmap(jpg);
    if (!live || live.dw !== meta.dw || live.dh !== meta.dh) {
      if (live) { world.remove(live.mesh); live.dispose(); }
      live = new Surface(meta.dw, meta.dh, 1.0);
      world.add(live.mesh);
    }
    if (lastPose && Math.abs(lastPose.yaw - meta.yaw) < 0.015 && Math.abs(lastPose.pitch - meta.pitch) < 0.015) {
      stableFrames++;
    } else stableFrames = 0;
    lastPose = { yaw: meta.yaw, pitch: meta.pitch };
    if (memoryOn && live.bitmap && meta.has_depth) rememberLive();
    live.update(meta, depth, bitmap);
    updateSkeleton(meta);
    floor.position.y = -(meta.camera_height || 1.3);
    state.lastHud = meta.hud || {};
    if (!state.initDone) { state.headControl = meta.head_control_default; state.initDone = true; }
    if (meta.rtt_ms != null) state.latency = meta.rtt_ms;
    frames++;
    const now = performance.now();
    if (now - lastFpsT > 1000) { state.fps = frames * 1000 / (now - lastFpsT); frames = 0; lastFpsT = now; }
    if (!renderer.xr.isPresenting) msg(`live · ${state.fps.toFixed(0)} fps`);
  } else if (type === 2) {
    const bitmap = await createImageBitmap(new Blob([new Uint8Array(buf, off)], { type: 'image/jpeg' }));
    if (gripTex.image && gripTex.image.close) gripTex.image.close();
    gripTex.image = bitmap; gripTex.needsUpdate = true;
    gripper.scale.set(1, (meta.h / meta.w) / (0.28 / 0.5), 1);
    gripper.visible = true;
  } else if (type === 3) {
    const bitmap = await createImageBitmap(new Blob([new Uint8Array(buf, off)], { type: 'image/jpeg' }));
    if (navTex.image && navTex.image.close) navTex.image.close();
    navTex.image = bitmap; navTex.needsUpdate = true;
    ensureNav(meta);
    nav.visible = $('navtoggle') ? $('navtoggle').checked : true;
    if (live) { nav.matrix.copy(poseMatrix(live.yaw, live.pitch)); nav.matrixWorldNeedsUpdate = true; }
  }
}

// ------------------------------------------------------------------ head-look telepresence + controls
const euler = new THREE.Euler(0, 0, 0, 'YXZ');
let lastLookSent = 0;
function viewerYawPitch() {
  const cam = renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;
  euler.setFromQuaternion(cam.quaternion, 'YXZ');
  return { yaw: euler.y, pitch: euler.x };
}
function sendLook(force = false) {
  const now = performance.now();
  if (!ws || ws.readyState !== 1 || (!force && now - lastLookSent < 100)) return;
  lastLookSent = now;
  const active = renderer.xr.isPresenting && state.headControl;
  const { yaw, pitch } = viewerYawPitch();
  ws.send(JSON.stringify({ type: 'look', yaw: yaw - world.rotation.y, pitch, active }));
}
function recenter() {
  // bring what the robot currently sees in front of you
  const { yaw } = viewerYawPitch();
  world.rotation.y = yaw - (live ? live.yaw : 0);
  ui.rotation.y = yaw;
}

const prevButtons = new Map();
function pollGamepads() {
  const session = renderer.xr.getSession();
  if (!session) return;
  for (const src of session.inputSources) {
    const gp = src.gamepad;
    if (!gp) continue;
    const prev = prevButtons.get(src) || [];
    const pressed = (i) => gp.buttons[i] && gp.buttons[i].pressed && !prev[i];
    if (pressed(4)) { state.headControl = !state.headControl; sendLook(true); }   // A / X
    if (pressed(5)) { memoryOn = !memoryOn; }                                      // B / Y
    prevButtons.set(src, gp.buttons.map((b) => b.pressed));
  }
}
for (const i of [0, 1]) {
  renderer.xr.getController(i).addEventListener('selectstart', recenter);
  renderer.xr.getController(i).addEventListener('squeezestart', ackAlarm);
}

renderer.xr.addEventListener('sessionstart', () => { setTimeout(recenter, 300); });
renderer.xr.addEventListener('sessionend', () => { sendLook(true); });

// desktop / 2-D preview: drag to look around
let drag = null;
renderer.domElement.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY }; });
window.addEventListener('pointerup', () => { drag = null; });
window.addEventListener('pointermove', (e) => {
  if (!drag || renderer.xr.isPresenting) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag = { x: e.clientX, y: e.clientY };
  euler.setFromQuaternion(camera.quaternion, 'YXZ');
  euler.y -= dx * 0.004;
  euler.x = THREE.MathUtils.clamp(euler.x - dy * 0.004, -1.4, 1.4);
  camera.quaternion.setFromEuler(euler);
});
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => {
  pollGamepads();
  sendLook();
  updateMemoryVisibility();
  renderer.render(scene, camera);
});

// ------------------------------------------------------------------ Enter VR
async function setupVR() {
  const btn = $('enter');
  if (!window.isSecureContext) { msg('VR needs https:// - open this page with https://'); return; }
  if (!navigator.xr || !(await navigator.xr.isSessionSupported('immersive-vr'))) {
    msg('WebXR VR not available in this browser (use the Meta Quest browser). 2-D preview below.');
    return;
  }
  btn.disabled = false;
  btn.onclick = async () => {
    if (renderer.xr.isPresenting) { renderer.xr.getSession().end(); return; }
    const session = await navigator.xr.requestSession('immersive-vr', { optionalFeatures: ['local-floor'] });
    await renderer.xr.setSession(session);
    btn.textContent = 'Exit VR';
    session.addEventListener('end', () => { btn.textContent = 'Enter VR'; });
  };
}
setupVR();
connect();

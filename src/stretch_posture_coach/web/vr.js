// Stretch telepresence + expert teleoperation for the Meta Quest browser (WebXR + three.js).
//
// Everything lives in ONE registered map frame (z up, from lidar SLAM + head joints + floor plane):
//   room scan (persistent voxels, incremental deltas) · live person surface + 3-D skeleton · lidar scan +
//   occupancy grid · robot avatar from joint states · posture heat map · captions pinned where things happened.
// `stage` maps that frame into the headset's local-floor space, so the map floor IS your floor.
// The expert can take control (PIN): arm follows the right hand while GRIP is held, A/B gripper with haptic
// feedback, left stick drives the base.

import * as THREE from 'three';

const $ = (id) => document.getElementById(id);
const msg = (s) => { $('msg').textContent = s; };

// ------------------------------------------------------------------ renderer / scene
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
if (renderer.xr.setFoveation) renderer.xr.setFoveation(0.5);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0c10);
const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.03, 80);
camera.position.set(0, 1.6, 0);
scene.add(camera);
scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.2));

// map frame (x fwd, y left, z up) -> three (x right, y up, -z fwd)
const M_MAP = new THREE.Matrix4().set(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 1);
const stage = new THREE.Group();          // children are in MAP coordinates
stage.matrixAutoUpdate = false;
scene.add(stage);
const place = { yaw: 0, tx: 0, tz: 0 };   // stage placement in the headset space
function updateStage() {
  const m = new THREE.Matrix4().makeTranslation(place.tx, 0, place.tz);
  m.multiply(new THREE.Matrix4().makeRotationY(place.yaw)).multiply(M_MAP);
  stage.matrix.copy(m);
  stage.matrixWorldNeedsUpdate = true;
}
updateStage();
const ui = new THREE.Group();             // panels in front of the user
scene.add(ui);

const grid = new THREE.GridHelper(30, 30, 0x3a3f4a, 0x1d2026);
grid.rotation.x = Math.PI / 2;             // grid in the map's floor plane
grid.material.transparent = true; grid.material.opacity = 0.35; grid.material.depthWrite = false;
stage.add(grid);

const C_OPT = new THREE.Matrix4().makeScale(1, -1, -1); // three-camera convention -> optical
const S = {                                 // client state
  connected: false, fps: 0, latency: 0, lastHud: {}, headControl: true, overlay: true, ceilingOn: true,
  alarm: { active: false }, registered: '-', world: null, robot: null, T: null, ceiling: 2.2,
  teleop: 'none', teleopOn: false, tfb: null, wrist: true, viewMode: 0,
};

// ------------------------------------------------------------------ room scan (voxels)
const CAP = 500000;
const mapPos = new Float32Array(CAP * 3).fill(1e6);
const mapCol = new Uint8Array(CAP * 3);
const mapGeo = new THREE.BufferGeometry();
const posAttr = new THREE.BufferAttribute(mapPos, 3); posAttr.setUsage(THREE.DynamicDrawUsage);
const colAttr = new THREE.BufferAttribute(mapCol, 3, true); colAttr.setUsage(THREE.DynamicDrawUsage);
mapGeo.setAttribute('position', posAttr);
mapGeo.setAttribute('color', colAttr);
mapGeo.setDrawRange(0, 0);
mapGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
const mapMat = new THREE.ShaderMaterial({
  uniforms: { voxel: { value: 0.05 }, heightPx: { value: 1000 }, ceiling: { value: 2.2 }, ceilingOn: { value: 1 } },
  vertexShader: /* glsl */`
    uniform float voxel; uniform float heightPx; uniform float ceiling; uniform float ceilingOn;
    attribute vec3 color; varying vec3 vColor; varying float vKeep;
    void main() {
      vColor = pow(color, vec3(2.2));   // sRGB bytes -> linear (re-encoded below)
      vKeep = (ceilingOn > 0.5 && position.z > ceiling) || position.z > 1e5 ? 0.0 : 1.0;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = clamp(voxel * 1.3 * heightPx * 0.5 * projectionMatrix[1][1] / max(-mv.z, 0.05), 1.0, 64.0);
    }`,
  fragmentShader: /* glsl */`
    varying vec3 vColor; varying float vKeep;
    void main() {
      if (vKeep < 0.5) discard;
      vec2 d = gl_PointCoord - 0.5;
      if (dot(d, d) > 0.25) discard;      // round splats
      gl_FragColor = vec4(vColor, 1.0);
      #include <colorspace_fragment>
    }`,
});
const mapPoints = new THREE.Points(mapGeo, mapMat);
mapPoints.frustumCulled = false;
stage.add(mapPoints);
const slotOf = new Map();
const freeSlots = [];
let maxSlot = 0;
const vkey = (x, y, z) => ((x + 32768) * 65536 + (y + 32768)) * 65536 + (z + 32768);

function clearMap() {
  slotOf.clear(); freeSlots.length = 0; maxSlot = 0;
  mapPos.fill(1e6); posAttr.needsUpdate = true; mapGeo.setDrawRange(0, 0);
}
function applyMapDelta(meta, buf, off) {
  if (meta.full) clearMap();
  const n = meta.n, m = meta.m, vox = meta.voxel;
  mapMat.uniforms.voxel.value = vox;
  const idx = new Int16Array(buf.slice(off, off + n * 6)); off += n * 6;
  const rgb = new Uint8Array(buf, off, n * 3); off += n * 3;
  const del = new Int16Array(buf.slice(off, off + m * 6));
  for (let i = 0; i < m; i++) {
    const k = vkey(del[3 * i], del[3 * i + 1], del[3 * i + 2]);
    const s = slotOf.get(k);
    if (s === undefined) continue;
    slotOf.delete(k); freeSlots.push(s);
    mapPos[3 * s] = mapPos[3 * s + 1] = mapPos[3 * s + 2] = 1e6;
  }
  for (let i = 0; i < n; i++) {
    const x = idx[3 * i], y = idx[3 * i + 1], z = idx[3 * i + 2];
    const k = vkey(x, y, z);
    let s = slotOf.get(k);
    if (s === undefined) {
      s = freeSlots.length ? freeSlots.pop() : maxSlot++;
      if (s >= CAP) { maxSlot = CAP; continue; }
      slotOf.set(k, s);
      mapPos[3 * s] = (x + 0.5) * vox; mapPos[3 * s + 1] = (y + 0.5) * vox; mapPos[3 * s + 2] = (z + 0.5) * vox;
    }
    mapCol[3 * s] = rgb[3 * i]; mapCol[3 * s + 1] = rgb[3 * i + 1]; mapCol[3 * s + 2] = rgb[3 * i + 2];
  }
  posAttr.needsUpdate = true; colAttr.needsUpdate = true;
  mapGeo.setDrawRange(0, maxSlot);
  S.mapVoxels = slotOf.size;
}

// ------------------------------------------------------------------ live RGB-D surface (registered)
const VS = /* glsl */`
  uniform sampler2D depthTex; uniform vec4 intr; uniform vec2 texel; uniform float maxRange; uniform float jump;
  uniform float personOnly;
  varying vec2 vUv; varying float vValid;
  void main() {
    vec2 st = uv;
    float raw = texture2D(depthTex, st).r;
    float z = abs(raw);
    float zr = abs(texture2D(depthTex, st + vec2(texel.x, 0.0)).r);
    float zd = abs(texture2D(depthTex, st + vec2(0.0, texel.y)).r);
    float zl = abs(texture2D(depthTex, st - vec2(texel.x, 0.0)).r);
    float zu = abs(texture2D(depthTex, st - vec2(0.0, texel.y)).r);
    float m = max(max(abs(z - zr), abs(z - zd)), max(abs(z - zl), abs(z - zu)));
    bool person = raw < 0.0;
    vValid = (z > 0.15 && z < maxRange && m < jump * z && (personOnly < 0.5 || person)) ? 1.0 : 0.0;
    vec3 p = vec3((st.x - intr.z) / intr.x * z, -(st.y - intr.w) / intr.y * z, -z);
    vUv = st;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }`;
const FS = /* glsl */`
  uniform sampler2D colorTex; varying vec2 vUv; varying float vValid;
  void main() {
    if (vValid < 0.99) discard;
    gl_FragColor = vec4(texture2D(colorTex, vUv).rgb, 1.0);
    #include <colorspace_fragment>
  }`;
function gridGeometry(dw, dh) {
  const pos = new Float32Array(dw * dh * 3), uv = new Float32Array(dw * dh * 2);
  for (let j = 0; j < dh; j++) for (let i = 0; i < dw; i++) {
    const k = j * dw + i; uv[2 * k] = (i + 0.5) / dw; uv[2 * k + 1] = (j + 0.5) / dh;
  }
  const idx = [];
  for (let j = 0; j < dh - 1; j++) for (let i = 0; i < dw - 1; i++) {
    const a = j * dw + i, b = a + 1, c = a + dw, d = c + 1; idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 100);
  return g;
}
class Surface {
  constructor(dw, dh) {
    this.dw = dw; this.dh = dh;
    this.depth = new THREE.DataTexture(new Float32Array(dw * dh), dw, dh, THREE.RedFormat, THREE.FloatType);
    this.depth.minFilter = this.depth.magFilter = THREE.NearestFilter;
    this.color = new THREE.Texture(); this.color.flipY = false; this.color.colorSpace = THREE.SRGBColorSpace;
    this.color.minFilter = THREE.LinearFilter; this.color.generateMipmaps = false;
    this.material = new THREE.ShaderMaterial({
      vertexShader: VS, fragmentShader: FS,
      uniforms: {
        depthTex: { value: this.depth }, colorTex: { value: this.color }, intr: { value: new THREE.Vector4() },
        texel: { value: new THREE.Vector2(1 / dw, 1 / dh) }, maxRange: { value: 6 }, jump: { value: 0.06 },
        personOnly: { value: 1 },
      },
    });
    this.mesh = new THREE.Mesh(gridGeometry(dw, dh), this.material);
    this.mesh.matrixAutoUpdate = false; this.mesh.frustumCulled = false;
  }
  update(meta, depthMM, bitmap) {
    const d = this.depth.image.data;
    for (let i = 0; i < d.length; i++) {
      const v = depthMM[i], z = (v & 0x7fff) * 0.001;
      d[i] = v & 0x8000 ? -z : z;             // negative = person pixel
    }
    this.depth.needsUpdate = true;
    if (this.color.image && this.color.image.close) this.color.image.close();
    this.color.image = bitmap; this.color.needsUpdate = true;
    this.material.uniforms.intr.value.set(...meta.intr);
    this.material.uniforms.maxRange.value = meta.max_range || 6;
  }
}
let live = null;
function camMatrix(meta) {   // map <- three-camera convention
  if (meta.T) return new THREE.Matrix4().set(...meta.T).multiply(C_OPT);
  // no registration (shouldn't happen): head yaw/pitch at camera height
  const m = new THREE.Matrix4().makeTranslation(0, 0, meta.camera_height || 1.3);
  return m.multiply(new THREE.Matrix4().makeRotationZ(meta.yaw)).multiply(new THREE.Matrix4().makeRotationY(-meta.pitch))
    .multiply(new THREE.Matrix4().set(0, 0, -1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1));
}

// ------------------------------------------------------------------ skeleton (3-D, colored by risk)
const LEVEL_COLORS = [new THREE.Color(0x50c850), new THREE.Color(0xffc800), new THREE.Color(0xeb3c3c)];
const NEUTRAL = new THREE.Color(0x3cb4ff);
const skelGeo = new THREE.BufferGeometry();
skelGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(64 * 3), 3));
skelGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(64 * 3), 3));
const skeleton = new THREE.LineSegments(skelGeo, new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false }));
skeleton.matrixAutoUpdate = false; skeleton.renderOrder = 10; skeleton.frustumCulled = false;
stage.add(skeleton);
const REGION = (a, b) => {
  const s = new Set([a, b]);
  if ([0, 7, 8].some((i) => s.has(i))) return 'neck';
  if ([13, 15].some((i) => s.has(i))) return 'l_arm';
  if ([14, 16].some((i) => s.has(i))) return 'r_arm';
  return 'trunk';
};
function updateSkeleton(meta, M) {
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
      col[3 * n] = c.r; col[3 * n + 1] = c.g; col[3 * n + 2] = c.b; n++;
    }
  }
  skelGeo.setDrawRange(0, n);
  skelGeo.attributes.position.needsUpdate = true; skelGeo.attributes.color.needsUpdate = true;
  skeleton.matrix.copy(M); skeleton.matrixWorldNeedsUpdate = true;
  skeleton.visible = n > 0;
}

// ------------------------------------------------------------------ lidar scan + occupancy grid
const lidarGeo = new THREE.BufferGeometry();
lidarGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(4096 * 3), 3));
lidarGeo.setDrawRange(0, 0);
const lidarPts = new THREE.Points(lidarGeo, new THREE.PointsMaterial({ color: 0xeb6834, size: 0.035 }));
lidarPts.frustumCulled = false;
stage.add(lidarPts);
let occMesh = null;
async function applyOccupancy(meta, png) {
  const bmp = await createImageBitmap(new Blob([png], { type: 'image/png' }));
  const cv = document.createElement('canvas'); cv.width = meta.w; cv.height = meta.h;
  const cx = cv.getContext('2d'); cx.drawImage(bmp, 0, 0); bmp.close();
  const im = cx.getImageData(0, 0, meta.w, meta.h), d = im.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = d[i];
    if (v < 64) { d[i] = 0xc3; d[i + 1] = 0xc2; d[i + 2] = 0xb7; d[i + 3] = 230; }      // wall / obstacle
    else if (v > 192) { d[i] = 0x2a; d[i + 1] = 0x30; d[i + 2] = 0x3a; d[i + 3] = 70; }  // free floor
    else d[i + 3] = 0;                                                                   // unknown
  }
  cx.putImageData(im, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.flipY = false; tex.magFilter = THREE.NearestFilter; tex.colorSpace = THREE.SRGBColorSpace;
  if (occMesh) { stage.remove(occMesh); occMesh.geometry.dispose(); occMesh.material.map.dispose(); occMesh.material.dispose(); }
  const w = meta.w * meta.res, h = meta.h * meta.res;
  occMesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
  const uv = occMesh.geometry.attributes.uv;        // canvas row 0 = min y
  for (let i = 0; i < uv.count; i++) uv.setY(i, 1 - uv.getY(i));
  occMesh.position.set(meta.origin[0] + w / 2, meta.origin[1] + h / 2, 0.004);
  occMesh.renderOrder = -2;
  stage.add(occMesh);
}

// ------------------------------------------------------------------ posture heat map + captions
const heatGroup = new THREE.Group(); stage.add(heatGroup);
let heatMesh = null;
S.heatLegend = null;
function applyHeat(h) {
  if (heatMesh) { heatGroup.remove(heatMesh); heatMesh.geometry.dispose(); heatMesh.material.dispose(); heatMesh = null; }
  S.heatLegend = h.legend;
  const cells = h.cells || [];
  if (!cells.length) return;
  heatMesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(h.cell * 0.94, h.cell * 0.94),
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.75, depthWrite: false }), cells.length);
  const m = new THREE.Matrix4(), c = new THREE.Color();
  cells.forEach((cell, i) => {
    m.makeTranslation(cell.x, cell.y, 0.07);   // just above the floor voxels
    heatMesh.setMatrixAt(i, m);
    heatMesh.setColorAt(i, c.set(cell.color));
  });
  heatMesh.renderOrder = -1;
  heatGroup.add(heatMesh);
}
const captionGroup = new THREE.Group(); stage.add(captionGroup);
const captionObjs = new Map();
function captionCanvas(c) {
  const cv = document.createElement('canvas'); cv.width = 640; cv.height = 190;
  const x = cv.getContext('2d');
  x.fillStyle = 'rgba(16,17,22,0.92)'; x.beginPath(); x.roundRect(0, 0, 640, 190, 22); x.fill();
  x.fillStyle = c.color; x.beginPath(); x.roundRect(0, 0, 18, 190, [22, 0, 0, 22]); x.fill();
  x.beginPath(); x.arc(70, 62, 30, 0, 2 * Math.PI); x.fill();
  x.fillStyle = c.level === 'warning' ? '#0b0b0b' : '#ffffff';
  x.font = 'bold 36px system-ui'; x.textAlign = 'center'; x.fillText(c.icon || '•', 70, 75);
  x.textAlign = 'left'; x.fillStyle = '#f0efec'; x.font = 'bold 38px system-ui';
  x.fillText(c.title.slice(0, 26), 118, 74);
  x.fillStyle = '#c3c2b7'; x.font = '28px system-ui';
  const words = String(c.text || '').split(' '); let line = '', y = 124;
  for (const w of words) {
    const t = line ? `${line} ${w}` : w;
    if (x.measureText(t).width > 590 && line) { x.fillText(line, 30, y); y += 34; line = w; if (y > 170) break; }
    else line = t;
  }
  if (line && y <= 170) x.fillText(line, 30, y);
  if (c.t) {
    x.fillStyle = '#898781'; x.font = '24px system-ui'; x.textAlign = 'right';
    x.fillText(new Date(c.t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), 620, 40);
  }
  return cv;
}
function applyCaptions(items) {
  const keep = new Set(items.filter((c) => c.pos).map((c) => c.id));
  for (const [id, o] of captionObjs) {
    if (!keep.has(id)) { captionGroup.remove(o.group); o.tex.dispose(); o.sprite.material.dispose(); captionObjs.delete(id); }
  }
  items.forEach((c) => {
    if (!c.pos || captionObjs.has(c.id)) return;
    const tex = new THREE.CanvasTexture(captionCanvas(c)); tex.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
    sprite.scale.set(0.64, 0.19, 1); sprite.renderOrder = 30;
    const g = new THREE.Group();
    const top = Math.max(c.pos[2] + 0.75, 1.9) + (c.lift || 0) * 0.8;
    sprite.position.set(c.pos[0], c.pos[1], top);
    const lg = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(c.pos[0], c.pos[1], 0.02),
      new THREE.Vector3(c.pos[0], c.pos[1], top - 0.1)]);
    const line = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: c.color, transparent: true, opacity: 0.8 }));
    const dot = new THREE.Mesh(new THREE.CircleGeometry(0.09, 24), new THREE.MeshBasicMaterial({ color: c.color, transparent: true, opacity: 0.85 }));
    dot.position.set(c.pos[0], c.pos[1], 0.02);
    g.add(sprite, line, dot);
    captionGroup.add(g);
    captionObjs.set(c.id, { group: g, sprite, tex, c, born: performance.now() });
  });
  S.captions = items;
}
function animateCaptions(now) {
  const ids = [...captionObjs.keys()].sort((a, b) => b - a);
  ids.forEach((id, rank) => {
    const o = captionObjs.get(id);
    const ageS = o.c.t ? Date.now() / 1000 - o.c.t : 0;
    let op = rank < 4 ? 1 : 0.55;
    if (ageS > 180 && o.c.level !== 'critical') op *= 0.6;
    if (o.c.kind === 'fall_alarm') op = 0.65 + 0.35 * Math.sin(now / 160);
    o.sprite.material.opacity = op;
    const s = rank === 0 ? 1.25 : 1.0;
    o.sprite.scale.set(0.64 * s, 0.19 * s, 1);
  });
}

// ------------------------------------------------------------------ robot avatar (approximate geometry)
const avatar = new THREE.Group(); avatar.matrixAutoUpdate = false; stage.add(avatar);
const amat = new THREE.MeshStandardMaterial({ color: 0x9aa0aa, transparent: true, opacity: 0.55, roughness: 0.7 });
const box = (w, d, h, color) => new THREE.Mesh(new THREE.BoxGeometry(w, d, h), color ? new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.7 }) : amat);
const MAST = [-0.07, 0.12];  // mast position in base_link (approx.)
const aBase = box(0.34, 0.33, 0.16); aBase.position.set(-0.05, 0, 0.09); avatar.add(aBase);
const aMast = box(0.07, 0.07, 1.1); aMast.position.set(MAST[0], MAST[1], 0.17 + 0.55); avatar.add(aMast);
const aArm = box(0.06, 1, 0.05, 0xcfd3da); avatar.add(aArm);
const aWrist = new THREE.Group(); avatar.add(aWrist);
const aGrip = box(0.06, 0.05, 0.16, 0x2a78d6); aGrip.position.set(0, 0, -0.07); aWrist.add(aGrip);
const fingers = [box(0.015, 0.02, 0.09, 0xeeeeee), box(0.015, 0.02, 0.09, 0xeeeeee)];
fingers.forEach((f) => { f.position.z = -0.17; aWrist.add(f); });
const aHead = box(0.05, 0.12, 0.04, 0x3a3f4a); aHead.matrixAutoUpdate = false; stage.add(aHead);
function updateAvatar(r, T) {
  if (!r) return;
  const [x, y, th] = r.base || [0, 0, 0];
  avatar.matrix.makeTranslation(x, y, 0).multiply(new THREE.Matrix4().makeRotationZ(th));
  avatar.matrixWorldNeedsUpdate = true;
  const j = r.joints || {};
  const armVisible = r.has_arm && j.lift != null;
  aArm.visible = aWrist.visible = armVisible;
  if (armVisible) {
    const ext = (j.arm || 0) + 0.25, z = (j.lift || 0) + 0.18;
    aArm.scale.y = ext; aArm.position.set(MAST[0], MAST[1] - 0.05 - ext / 2, z);
    aWrist.position.set(MAST[0], MAST[1] - 0.05 - ext, z - 0.02);
    aWrist.rotation.set(0, j.wrist_pitch || 0, j.wrist_yaw || 0, 'ZYX');
    const g = THREE.MathUtils.clamp(((j.gripper ?? 0) + 100) / 200, 0, 1);
    fingers[0].position.x = -0.012 - 0.04 * g; fingers[1].position.x = 0.012 + 0.04 * g;
  }
  if (T) { aHead.matrix.copy(T); aHead.matrixWorldNeedsUpdate = true; }
}

// ------------------------------------------------------------------ panels: HUD, gripper cam, wide cam
function canvasPanel(wPx, hPx, wM) {
  const canvas = document.createElement('canvas'); canvas.width = wPx; canvas.height = hPx;
  const tex = new THREE.CanvasTexture(canvas); tex.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(wM, wM * hPx / wPx),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthTest: false }));
  mesh.renderOrder = 40;
  return { canvas, ctx: canvas.getContext('2d'), tex, mesh };
}
const hud = canvasPanel(1024, 760, 0.62);
function placePanel(mesh, x, y, z) { mesh.position.set(x, y, z); mesh.lookAt(0, y + 0.0, 0.0); }
placePanel(hud.mesh, 0.62, 1.15, -1.0);
ui.add(hud.mesh);
function imagePanel(wM, hM) {
  const tex = new THREE.Texture(); tex.colorSpace = THREE.SRGBColorSpace; tex.flipY = false; tex.generateMipmaps = false;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(wM, hM), new THREE.MeshBasicMaterial({ map: tex, depthTest: false }));
  const uv = mesh.geometry.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setY(i, 1 - uv.getY(i));
  mesh.renderOrder = 40; mesh.visible = false;
  return { tex, mesh, w: wM, h: hM };
}
const gripPanel = imagePanel(0.46, 0.26); placePanel(gripPanel.mesh, -0.62, 1.0, -1.0); ui.add(gripPanel.mesh);
const navPanel = imagePanel(0.46, 0.34); placePanel(navPanel.mesh, -0.62, 1.38, -1.05); ui.add(navPanel.mesh);
const forceBar = new THREE.Mesh(new THREE.PlaneGeometry(0.46, 0.025), new THREE.MeshBasicMaterial({ color: 0xec835a, depthTest: false }));
forceBar.renderOrder = 41; gripPanel.mesh.add(forceBar); forceBar.position.set(0, -0.145, 0.001);

function wrap(ctx, text, x, y, maxW, lh, maxLines) {
  const words = String(text).split(' '); let line = '', n = 0;
  for (const w of words) {
    const t = line ? `${line} ${w}` : w;
    if (ctx.measureText(t).width > maxW && line) { ctx.fillText(line, x, y); y += lh; line = w; if (++n >= maxLines) return y; }
    else line = t;
  }
  if (line) { ctx.fillText(line, x, y); y += lh; }
  return y;
}
function drawHud() {
  const { ctx, canvas } = hud, h = S.lastHud || {};
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'rgba(16,17,22,0.88)'; ctx.beginPath(); ctx.roundRect(0, 0, canvas.width, canvas.height, 28); ctx.fill();
  ctx.fillStyle = '#ffd27a'; ctx.font = 'bold 38px system-ui'; ctx.fillText('STRETCH POSTURE COACH', 32, 56);
  ctx.font = '26px system-ui'; ctx.fillStyle = '#9aa0aa';
  const w = S.world || {};
  const lid = w.lidar && w.lidar.enabled ? `lidar SLAM ${w.lidar.score}` : 'no lidar: odometry';
  ctx.fillText(`${S.connected ? 'live' : 'offline'} · ${S.fps.toFixed(0)} fps · ${S.latency.toFixed(0)} ms · ${lid} · ${(S.mapVoxels || 0).toLocaleString()} voxels`, 32, 94);
  let y = 146;
  const risk = (s) => (s == null ? '#ccc' : s >= 5 ? '#eb3c3c' : s >= 3 ? '#ffc800' : '#50c850');
  if (h.present === false) { ctx.fillStyle = '#ffc800'; ctx.font = '32px system-ui'; ctx.fillText('No person in view', 32, y); y += 48; }
  else if (h.present) {
    ctx.font = '32px system-ui'; ctx.fillStyle = '#eee'; ctx.fillText(`Posture: ${h.posture || '?'}`, 32, y);
    if (h.rula) { ctx.fillStyle = risk(h.rula.score); ctx.fillText(`RULA ${h.rula.score} (${h.rula.risk})`, 420, y); }
    y += 44;
    if (h.reba) { ctx.fillStyle = risk(h.reba.score - 2); ctx.fillText(`REBA ${h.reba.score} (${h.reba.risk})`, 420, y); }
    ctx.fillStyle = '#eee'; ctx.fillText(`Neck ${h.neck_flexion ?? '-'}°  Trunk ${h.trunk_flexion ?? '-'}°`, 32, y); y += 44;
    ctx.fillText(`L5/S1 ${h.l5s1_compression_n ?? '-'} N`, 32, y); y += 46;
    ctx.fillStyle = '#eb6b6b';
    for (const issue of (h.issues || []).slice(0, 2)) { ctx.fillText(`• ${issue}`, 32, y); y += 40; }
  }
  const f = h.follow;
  if (f && f.distance_m != null) {
    ctx.fillStyle = f.distance_m >= 2 && f.distance_m <= 3 ? '#50c850' : '#ffc800'; ctx.font = '28px system-ui';
    ctx.fillText(`Distance ${f.distance_m.toFixed(2)} m${f.enabled ? ` · follow: ${f.action}` : ''}`, 32, y); y += 42;
  }
  if (h.last_message) { ctx.fillStyle = '#ffd27a'; ctx.font = 'italic 26px system-ui'; y = wrap(ctx, `“${h.last_message}”`, 32, y + 4, canvas.width - 64, 34, 2); }
  // expert control
  const t = S.tfb, owned = S.teleop === 'granted';
  if (owned) {
    y += 10;
    ctx.fillStyle = S.teleopOn ? '#3987e5' : '#52514e'; ctx.fillRect(24, y - 30, canvas.width - 48, 4);
    ctx.font = 'bold 28px system-ui'; ctx.fillStyle = '#f0efec';
    ctx.fillText(S.teleopOn ? `EXPERT CONTROL · ${S.handMode || 'controllers'}` : 'EXPERT CONTROL paused (left stick click)', 32, y + 6); y += 40;
    ctx.font = '25px system-ui'; ctx.fillStyle = '#c3c2b7';
    if (t && t.active) {
      ctx.fillText(`arm ${t.clutch ? 'FOLLOWING your hand' : 'hold right GRIP'} · wrist ${S.wrist ? 'on' : 'off'} · v ${t.v.toFixed(2)} m/s · w ${t.w.toFixed(2)} rad/s`, 32, y); y += 34;
      const g = (t.haptics && t.haptics.gripper) || 0;
      ctx.fillText('grip force', 32, y + 6);
      ctx.fillStyle = '#2c2c2a'; ctx.fillRect(170, y - 14, 380, 22);
      ctx.fillStyle = '#ec835a'; ctx.fillRect(170, y - 14, 380 * g, 22); y += 36;
      if (t.note) { ctx.fillStyle = '#fab219'; ctx.fillText(`▲ ${t.note}`, 32, y); y += 34; }
    }
  }
  // heat legend
  if (S.overlay && S.heatLegend) {
    const L = S.heatLegend, yy = canvas.height - 96, x0 = 32, wL = 300;
    L.ramp.forEach((c, i) => { ctx.fillStyle = c; ctx.fillRect(x0 + i * wL / L.ramp.length, yy, wL / L.ramp.length + 1, 16); });
    ctx.font = '20px system-ui'; ctx.fillStyle = '#c3c2b7';
    ctx.fillText('better', x0, yy + 40); ctx.textAlign = 'right'; ctx.fillText('worse posture (REBA)', x0 + wL, yy + 40); ctx.textAlign = 'left';
  }
  ctx.fillStyle = '#898781'; ctx.font = '22px system-ui';
  ctx.fillText(`X head follows you: ${S.headControl ? 'ON' : 'off'} · Y overlays: ${S.overlay ? 'ON' : 'off'} · L-trigger: snap to robot`, 32, canvas.height - 24);
  const al = S.alarm;
  if (al.active) {
    const on = Math.floor(performance.now() / 500) % 2 === 0 || al.state !== 'alarm';
    ctx.fillStyle = al.state === 'alarm' ? (on ? '#d03b3b' : '#7a1010') : '#c77700'; ctx.fillRect(0, 0, canvas.width, 150);
    ctx.fillStyle = '#fff'; ctx.font = 'bold 44px system-ui';
    ctx.fillText(al.state === 'alarm' ? 'FALL ALARM' : 'Possible fall - checking', 32, 60);
    ctx.font = '26px system-ui'; wrap(ctx, `${al.reason || ''} · hold LEFT grip = I'm OK`, 32, 104, canvas.width - 64, 32, 2);
  }
  hud.tex.needsUpdate = true;
  forceBar.scale.x = Math.max(0.001, (t && t.haptics && t.haptics.gripper) || 0);
  forceBar.position.x = -0.23 * (1 - forceBar.scale.x);
}
setInterval(drawHud, 200);

// ------------------------------------------------------------------ networking
let ws = null, frames = 0, lastFpsT = performance.now();
const decoder = new TextDecoder();
const send = (o) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); };
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { S.connected = true; clearMap(); msg('connected - waiting for video…'); if (S.teleop === 'granted' && $('pin').value) claim(); };
  ws.onclose = () => { S.connected = false; S.teleop = 'none'; updateTeleopUi(); msg('disconnected - retrying…'); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => { if (typeof ev.data === 'string') onText(JSON.parse(ev.data)); else onBinary(ev.data); };
}
function ackAlarm() { if (S.alarm.active) send({ type: 'alarm_ack' }); }
$('ack').onclick = ackAlarm;
function onAlarm(a) {
  S.alarm = a;
  const b = $('alarm'); b.style.display = a.active ? 'block' : 'none'; b.className = a.state === 'alarm' ? '' : 'check';
  $('atitle').textContent = a.state === 'alarm' ? 'FALL ALARM' : 'Possible fall - checking…';
  $('areason').textContent = a.reason || '';
  if (a.active) placePanel(hud.mesh, 0, 1.35, -0.85); else placePanel(hud.mesh, 0.62, 1.15, -1.0);
}
$('endsess').onclick = () => { if (confirm('End this session and save its report? Stretch keeps running.')) send({ type: 'end_session' }); };
function onText(s) {
  if (s.type === 'alarm') return onAlarm(s);
  if (s.type === 'report') { $('saved').innerHTML = s.status === 'ready' ? `Saved ${s.name} · <a href="${s.url}" target="_blank">report</a>` : `Saving ${s.name}…`; return; }
  if (s.type === 'captions') return applyCaptions(s.items || []);
  if (s.type === 'heat') return applyHeat(s);
  if (s.type === 'teleop') { S.teleop = s.state; if (s.state === 'granted') S.teleopOn = true; updateTeleopUi(s); return; }
  if (s.type === 'teleop_fb') { S.tfb = s; return hapticFeedback(s.haptics); }
  if (s.type === 'state' && s.state === 'ended') {
    msg('Session ended - opening the report…');
    const go = () => { location.href = s.report || '/sessions'; };
    const session = renderer.xr.getSession();
    if (session) session.end().then(go, go); else go();
  }
}
const ack = (type, extra) => send({ type: 'ack', t: type, ...(extra || {}) });
async function onBinary(buf) {
  const type = new DataView(buf).getUint8(0);
  try { await handleBinary(buf, type); } catch (e) { console.warn('frame', type, e); } finally { ack(type); }
}
async function handleBinary(buf, type) {
  const dv = new DataView(buf), jl = dv.getUint32(1, true);
  const meta = JSON.parse(decoder.decode(new Uint8Array(buf, 5, jl)));
  let off = 5 + jl;
  if (type === 1) {
    const il = dv.getUint32(off, true); off += 4;
    const bitmap = await createImageBitmap(new Blob([new Uint8Array(buf, off, il)], { type: 'image/jpeg' })); off += il;
    const depth = new Uint16Array(buf.slice(off));
    if (!live || live.dw !== meta.dw || live.dh !== meta.dh) {
      if (live) stage.remove(live.mesh);
      live = new Surface(meta.dw, meta.dh); stage.add(live.mesh);
    }
    live.update(meta, depth, bitmap);
    live.material.uniforms.personOnly.value = $('livebg').checked ? 0 : 1;
    const M = camMatrix(meta);
    live.mesh.matrix.copy(M); live.mesh.matrixWorldNeedsUpdate = true;
    updateSkeleton(meta, M);
    S.T = meta.T ? new THREE.Matrix4().set(...meta.T) : null;
    S.robot = meta.robot || null; S.world = meta.world || null; S.registered = meta.registered || '-';
    updateAvatar(S.robot, S.T);
    S.lastHud = meta.hud || {}; S.ceiling = meta.ceiling || 2.2; mapMat.uniforms.ceiling.value = S.ceiling;
    if (!S.initDone) { S.headControl = meta.head_control_default; S.initDone = true; setTimeout(() => snap(0), 50); }
    if (meta.rtt_ms != null) S.latency = meta.rtt_ms;
    frames++;
    const now = performance.now();
    if (now - lastFpsT > 1000) { S.fps = frames * 1000 / (now - lastFpsT); frames = 0; lastFpsT = now; }
    if (!renderer.xr.isPresenting) msg(`live · ${S.fps.toFixed(0)} fps · ${S.registered}`);
  } else if (type === 2 || type === 3) {
    const p = type === 2 ? gripPanel : navPanel;
    const bitmap = await createImageBitmap(new Blob([new Uint8Array(buf, off)], { type: 'image/jpeg' }));
    if (p.tex.image && p.tex.image.close) p.tex.image.close();
    p.tex.image = bitmap; p.tex.needsUpdate = true;
    p.mesh.scale.set(1, (meta.h / meta.w) / (p.h / p.w), 1); p.mesh.visible = true;
  } else if (type === 4) {
    applyMapDelta(meta, buf, off);
  } else if (type === 5) {
    const pts = new Float32Array(buf.slice(off, off + meta.n * 8));
    const a = lidarGeo.attributes.position.array, n = Math.min(meta.n, 4096);
    for (let i = 0; i < n; i++) { a[3 * i] = pts[2 * i]; a[3 * i + 1] = pts[2 * i + 1]; a[3 * i + 2] = 0.17; }
    lidarGeo.setDrawRange(0, n); lidarGeo.attributes.position.needsUpdate = true;
  } else if (type === 6) {
    await applyOccupancy(meta, new Uint8Array(buf, off));
  }
}

// ------------------------------------------------------------------ viewer placement
const euler = new THREE.Euler(0, 0, 0, 'YXZ');
function viewerPose() {
  const cam = renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;
  const p = new THREE.Vector3(), q = new THREE.Quaternion();
  cam.matrixWorld.decompose(p, q, new THREE.Vector3());
  euler.setFromQuaternion(q, 'YXZ');
  return { p, yaw: euler.y, pitch: euler.x };
}
// Put the viewer at map point (mx, my) facing map heading psi (rad).
function placeViewer(mx, my, psi) {
  const v = viewerPose();
  place.yaw = v.yaw - psi;
  const p3 = new THREE.Vector3(mx, my, 0).applyMatrix4(new THREE.Matrix4().makeRotationY(place.yaw).multiply(M_MAP));
  place.tx = v.p.x - p3.x; place.tz = v.p.z - p3.z;
  updateStage();
  ui.rotation.y = v.yaw;
}
// mode 0: "arm view" - stand at the mast facing the arm (hand forward = arm out); mode 1: robot head view
function snap(mode) {
  const r = S.robot, base = (r && r.base) || [0, 0, 0];
  const [x, y, th] = base, c = Math.cos(th), s = Math.sin(th);
  if (mode === 0) {
    const ox = MAST[0] - 0.05, oy = MAST[1] + 0.35;   // just behind the mast, on the side opposite the arm
    placeViewer(x + c * ox - s * oy, y + s * ox + c * oy, th - Math.PI / 2);
  } else {
    const T = S.T, hx = T ? T.elements[12] : x, hy = T ? T.elements[13] : y;
    const fwd = T ? new THREE.Vector3(0, 0, 1).transformDirection(T) : new THREE.Vector3(c, s, 0);
    placeViewer(hx, hy, Math.atan2(fwd.y, fwd.x));
  }
  S.viewMode = mode;
}

// ------------------------------------------------------------------ head look (telepresence)
let lastLookSent = 0;
function sendLook(force = false) {
  const now = performance.now();
  if (!force && now - lastLookSent < 100) return;
  lastLookSent = now;
  const active = renderer.xr.isPresenting && S.headControl;
  const { yaw, pitch } = viewerPose();
  send({ type: 'look', yaw: yaw - place.yaw, pitch, active, frame: 'map' });  // gaze heading in the map frame
}

// ------------------------------------------------------------------ expert teleop
function claim() { send({ type: 'teleop_claim', pin: $('pin').value.trim() }); }
$('claim').onclick = claim;
$('release').onclick = () => { send({ type: 'teleop_release' }); S.teleop = 'none'; S.teleopOn = false; updateTeleopUi(); };
function updateTeleopUi(s) {
  const st = S.teleop;
  const txt = { granted: 'you have control', denied: 'wrong PIN', busy: 'another expert has control',
    locked: `too many tries - wait ${s && s.retry_s ? s.retry_s : 60} s`, disabled: 'disabled on the robot', released: 'released', none: '' };
  $('tstate').textContent = txt[st] ?? st;
  $('tstate').style.color = st === 'granted' ? '#7fd07f' : '#eda100';
  $('release').style.display = st === 'granted' ? '' : 'none';
  $('claim').style.display = st === 'granted' ? 'none' : '';
}
const inputs = { left: null, right: null };       // XRInputSource per hand
const grips = [renderer.xr.getControllerGrip(0), renderer.xr.getControllerGrip(1)];
const hands = [renderer.xr.getHand(0), renderer.xr.getHand(1)];
grips.forEach((g) => scene.add(g));
hands.forEach((h) => scene.add(h));
const handOf = {};
[0, 1].forEach((i) => {
  grips[i].addEventListener('connected', (e) => { handOf[i] = e.data; });
  grips[i].addEventListener('disconnected', () => { delete handOf[i]; });
});
function sideIndex(side) { for (const i of [0, 1]) if (handOf[i] && handOf[i].handedness === side) return i; return -1; }
const prevBtn = { left: [], right: [] };
let leftGripSince = 0, lastTeleopSent = 0;
const tmpM = new THREE.Matrix4(), tmpP = new THREE.Vector3(), tmpQ = new THREE.Quaternion(), tmpS = new THREE.Vector3();
function toMap(objMatrixWorld) {
  tmpM.copy(stage.matrix).invert().multiply(objMatrixWorld);
  tmpM.decompose(tmpP, tmpQ, tmpS);
  return { p: [tmpP.x, tmpP.y, tmpP.z], q: [tmpQ.x, tmpQ.y, tmpQ.z, tmpQ.w] };
}
function jointDist(hand, a, b) {
  const ja = hand.joints[a], jb = hand.joints[b];
  if (!ja || !jb || !ja.visible || !jb.visible) return null;
  return ja.position.distanceTo(jb.position);
}
function pollInputs(now) {
  const session = renderer.xr.getSession();
  if (!session) return;
  const cmd = { type: 'teleop', drive: [0, 0], clutch: false, precision: false, wrist: S.wrist, gripper: null, hand: null, mode: 'controllers' };
  let haveRightCtl = false;
  for (const src of session.inputSources) {
    const side = src.handedness;
    if (side !== 'left' && side !== 'right') continue;
    inputs[side] = src;
    const gp = src.gamepad;
    if (src.hand || !gp) continue;
    const prev = prevBtn[side] || [];
    const b = (i) => gp.buttons[i] && gp.buttons[i].pressed;
    const edge = (i) => b(i) && !prev[i];
    if (side === 'left') {
      cmd.drive = [-(gp.axes[3] || 0), -(gp.axes[2] || 0)];
      if (edge(4)) { S.headControl = !S.headControl; sendLook(true); }     // X
      if (edge(5)) setOverlay(!S.overlay);                                  // Y
      if (edge(0)) snap(S.viewMode === 0 ? 1 : 0);                          // trigger
      if (edge(3) && S.teleop === 'granted') S.teleopOn = !S.teleopOn;     // stick click
      if (b(1)) { if (!leftGripSince) leftGripSince = now; if (now - leftGripSince > 800) { ackAlarm(); leftGripSince = now + 1e9; } }
      else leftGripSince = 0;
    } else {
      haveRightCtl = true;
      cmd.clutch = b(1);
      cmd.precision = gp.buttons[0] && gp.buttons[0].value > 0.5;
      if (edge(4)) cmd.gripper = 'open';                                    // A
      if (edge(5)) cmd.gripper = 'close';                                   // B
      if (edge(3)) { S.wrist = !S.wrist; cmd.wrist = S.wrist; }            // stick click
      const gi = sideIndex('right');
      if (gi >= 0) cmd.hand = toMap(grips[gi].matrixWorld);
    }
    prevBtn[side] = gp.buttons.map((x) => x.pressed);
  }
  if (!haveRightCtl) {   // bare hands: left pinch = clutch, right wrist = arm, right thumb-index = gripper
    const ri = sideIndex('right'), li = sideIndex('left');
    if (ri >= 0 && handOf[ri] && handOf[ri].hand) {
      const rh = hands[ri], wrist = rh.joints.wrist;
      if (wrist && wrist.visible) {
        cmd.mode = 'hands';
        cmd.hand = toMap(wrist.matrixWorld);
        const d = jointDist(rh, 'thumb-tip', 'index-finger-tip');
        if (d != null) cmd.aperture = THREE.MathUtils.clamp((d - 0.015) / 0.075, 0, 1);
      }
      if (li >= 0 && handOf[li] && handOf[li].hand) {
        const d = jointDist(hands[li], 'thumb-tip', 'index-finger-tip');
        cmd.clutch = d != null && d < 0.022;
      }
    }
  }
  S.handMode = cmd.mode;
  if (S.teleop === 'granted' && S.teleopOn && now - lastTeleopSent > 33) {
    lastTeleopSent = now; send(cmd);
  } else if (cmd.gripper && S.teleop === 'granted' && S.teleopOn) send(cmd);
}
function hapticFeedback(h) {
  if (!h) return;
  const src = inputs.right;
  const act = src && src.gamepad && src.gamepad.hapticActuators && src.gamepad.hapticActuators[0];
  const level = Math.max(h.gripper || 0, 0.6 * (h.arm || 0));
  try {
    if (h.contact && act) act.pulse(1.0, 60);
    else if (level > 0.02 && act) act.pulse(0.12 + 0.88 * level, 90);
  } catch (e) { /* not supported */ }
}

// ------------------------------------------------------------------ overlays / toggles
function setOverlay(on) { S.overlay = on; heatGroup.visible = captionGroup.visible = on; $('overlay').checked = on; }
$('overlay').onchange = (e) => setOverlay(e.target.checked);
$('ceil').onchange = (e) => { mapMat.uniforms.ceilingOn.value = e.target.checked ? 1 : 0; };

// desktop / 2-D preview: drag to look, WASD to walk, R to snap
let drag = null;
const keys = new Set();
renderer.domElement.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY }; });
window.addEventListener('pointerup', () => { drag = null; });
window.addEventListener('pointermove', (e) => {
  if (!drag || renderer.xr.isPresenting) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag = { x: e.clientX, y: e.clientY };
  euler.setFromQuaternion(camera.quaternion, 'YXZ');
  euler.y -= dx * 0.004; euler.x = THREE.MathUtils.clamp(euler.x - dy * 0.004, -1.4, 1.4);
  camera.quaternion.setFromEuler(euler);
});
function overview() {   // 2-D preview: bird's-eye view of the robot and the room
  const r = S.robot, [x, y] = (r && r.base) || [0, 0];
  const c = new THREE.Vector3(x, y, 0).applyMatrix4(stage.matrix);
  camera.position.set(c.x + 3.2, 4.2, c.z + 3.2);
  camera.lookAt(c.x, 0.4, c.z);
}
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase(); keys.add(k);
  if (k === 'r') snap(S.viewMode === 0 ? 1 : 0);
  if (k === 'o') overview();
});
window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
let prevT = performance.now();
renderer.setAnimationLoop((t) => {
  const now = performance.now(), dt = Math.min(0.1, (now - prevT) / 1000); prevT = now;
  if (!renderer.xr.isPresenting && keys.size) {
    const f = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion); f.y = 0; f.normalize();
    const r = new THREE.Vector3(-f.z, 0, f.x);
    const v = 1.5 * dt;
    if (keys.has('w')) camera.position.addScaledVector(f, v);
    if (keys.has('s')) camera.position.addScaledVector(f, -v);
    if (keys.has('d')) camera.position.addScaledVector(r, v);
    if (keys.has('a')) camera.position.addScaledVector(r, -v);
  }
  if (renderer.xr.isPresenting) {
    const xc = renderer.xr.getCamera();
    mapMat.uniforms.heightPx.value = (xc.cameras && xc.cameras[0] && xc.cameras[0].viewport) ? xc.cameras[0].viewport.w : 1800;
  } else mapMat.uniforms.heightPx.value = renderer.domElement.height;
  pollInputs(now);
  sendLook();
  animateCaptions(t);
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
    const session = await navigator.xr.requestSession('immersive-vr', { optionalFeatures: ['local-floor', 'hand-tracking'] });
    await renderer.xr.setSession(session);
    btn.textContent = 'Exit VR';
    session.addEventListener('end', () => { btn.textContent = 'Enter VR'; sendLook(true); });
    setTimeout(() => snap(S.viewMode), 300);
  };
}
setOverlay(true);
setupVR();
connect();
window.__vr = { S, stage, snap, overview, camera, applyMapDelta, slotOf };   // for debugging in the browser console

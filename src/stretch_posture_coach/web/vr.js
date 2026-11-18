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
const place = { yaw: 0, tx: 0, ty: 0, tz: 0 };   // stage placement in the headset space
function updateStage() {
  const m = new THREE.Matrix4().makeTranslation(place.tx, place.ty, place.tz);
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
  const st = S.status || {}, md = st.mode || {};
  const MODE_COL = { survey: '#3987e5', autonomous: '#50c850', teleop: '#ec835a', fall_response: '#d03b3b', safe_stop: '#d03b3b' };
  ctx.fillStyle = MODE_COL[md.mode] || '#ffd27a'; ctx.font = 'bold 36px system-ui';
  let title = (md.label || 'Stretch posture coach').toUpperCase();
  const sv = st.survey;
  if (md.mode === 'survey' && sv) title += `  ${sv.phase} ${sv.phase === 'sweep' ? `${sv.step + 1}/${sv.steps}` : ''}`;
  ctx.fillText(title, 32, 56);
  ctx.font = '24px system-ui'; ctx.fillStyle = '#c3c2b7'; ctx.textAlign = 'right';
  ctx.fillText(VIEW_NAMES[S.view] || '', canvas.width - 32, 56); ctx.textAlign = 'left';
  ctx.font = '26px system-ui'; ctx.fillStyle = '#9aa0aa';
  const w = S.world || {};
  const lid = w.lidar && w.lidar.enabled ? `lidar SLAM ${w.lidar.score}` : 'no lidar: odometry';
  const bar = S.barriers ? ` · ${S.barriers} barriers${S.barrierActive ? ' (ACTIVE)' : ''}` : '';
  ctx.fillText(`${S.connected ? 'live' : 'offline'} · ${S.fps.toFixed(0)} fps · ${S.latency.toFixed(0)} ms · ${lid} · ${(S.mapVoxels || 0).toLocaleString()} voxels${bar}`, 32, 94);
  let y = 146;
  const risk = (s) => (s == null ? '#ccc' : s >= 5 ? '#eb3c3c' : s >= 3 ? '#ffc800' : '#50c850');
  if (S.scene) { ctx.font = '24px system-ui'; ctx.fillStyle = '#9fc3f0'; ctx.fillText(S.scene.slice(0, 78), 32, y - 12); y += 28; }
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
  // voice relay
  const vo = S.voice;
  if (vo && performance.now() - (vo.t || 0) < 12000) {
    y += 6; ctx.font = '25px system-ui';
    if (vo.state === 'listening') { ctx.fillStyle = '#ec835a'; y = wrap(ctx, `● REC  ${vo.partial || 'listening…'}`, 32, y, canvas.width - 64, 32, 2); }
    else if (vo.state === 'transcribing' || vo.state === 'thinking') { ctx.fillStyle = '#c3c2b7'; y = wrap(ctx, `… ${vo.heard || vo.state}`, 32, y, canvas.width - 64, 32, 2); }
    else if (vo.state === 'spoken') { ctx.fillStyle = '#7fd07f'; y = wrap(ctx, `Stretch said: “${vo.said}”`, 32, y, canvas.width - 64, 32, 2); }
  }
  if (S.toast && performance.now() - S.toast.t < 3000) {
    ctx.fillStyle = 'rgba(57,135,229,0.9)'; ctx.fillRect(24, canvas.height - 150, canvas.width - 48, 44);
    ctx.fillStyle = '#fff'; ctx.font = '24px system-ui'; ctx.fillText(S.toast.text.slice(0, 70), 36, canvas.height - 120);
  }
  // heat legend
  if (S.overlay && S.heatLegend) {
    const L = S.heatLegend, yy = canvas.height - 96, x0 = 32, wL = 300;
    L.ramp.forEach((c, i) => { ctx.fillStyle = c; ctx.fillRect(x0 + i * wL / L.ramp.length, yy, wL / L.ramp.length + 1, 16); });
    ctx.font = '20px system-ui'; ctx.fillStyle = '#c3c2b7';
    ctx.fillText('better', x0, yy + 40); ctx.textAlign = 'right'; ctx.fillText('worse posture (REBA)', x0 + wL, yy + 40); ctx.textAlign = 'left';
  }
  ctx.fillStyle = '#898781'; ctx.font = '22px system-ui';
  ctx.fillText(`X head follows you: ${S.headControl ? 'ON' : 'off'} · Y camera (hold: overlays) · L-trigger: talk`, 32, canvas.height - 24);
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
  if (s.type === 'status') { S.status = s; return; }
  if (s.type === 'barriers') return applyBarriers(s);
  if (s.type === 'scene_graph') return applySceneGraph(s);
  if (s.type === 'voice') {
    S.voice = { ...(S.voice || {}), ...s, t: performance.now() };
    if (s.error) toast(s.error);
    if (s.state === 'spoken') { $('vstate').textContent = `Stretch said: “${s.said}”`; }
    else if (s.partial) $('vstate').textContent = `… ${s.partial}`;
    else $('vstate').textContent = s.state === 'listening' ? 'listening…' : (s.state || '');
    return;
  }
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
    if (!S.initDone) { S.headControl = meta.head_control_default; S.initDone = true; setTimeout(setReference, 50); }
    if (meta.rtt_ms != null) S.latency = meta.rtt_ms;
    frames++;
    const now = performance.now();
    if (now - lastFpsT > 1000) { S.fps = frames * 1000 / (now - lastFpsT); frames = 0; lastFpsT = now; }
    if (!renderer.xr.isPresenting) msg(`live · ${S.fps.toFixed(0)} fps · ${S.registered}`);
  } else if (type === 7) {   // gripper D405 RGB-D: panel + immersive surface for the gripper view
    const il = dv.getUint32(off, true); off += 4;
    const blob = new Blob([new Uint8Array(buf, off, il)], { type: 'image/jpeg' }); off += il;
    const depth = new Uint16Array(buf.slice(off));
    const bitmap = await createImageBitmap(blob);
    if (!gripLive || gripLive.dw !== meta.dw || gripLive.dh !== meta.dh) {
      if (gripLive) gripView.remove(gripLive.mesh);
      gripLive = new Surface(meta.dw, meta.dh);
      gripLive.material.uniforms.personOnly.value = 0;
      gripLive.material.uniforms.jump.value = 0.12;
      gripView.add(gripLive.mesh);
    }
    gripLive.update(meta, depth, bitmap);
    gripLive.mesh.matrix.identity(); gripLive.mesh.matrixWorldNeedsUpdate = true;
    const panelBmp = await createImageBitmap(blob);
    if (gripPanel.tex.image && gripPanel.tex.image.close) gripPanel.tex.image.close();
    gripPanel.tex.image = panelBmp; gripPanel.tex.needsUpdate = true;
    gripPanel.mesh.scale.set(1, (meta.h / meta.w) / (gripPanel.h / gripPanel.w), 1);
    gripPanel.mesh.visible = S.view !== 'gripper';
    S.gripperDepth = true;
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

// ------------------------------------------------------------------ views + reference ("0 rotation")
// When VR starts (or you recenter), your headset pose becomes the reference: looking straight ahead = the robot
// looking to the FRONT. From then on the robot head follows your rotation RELATIVE to that reference, and the
// world rides along with the robot so its forward stays in front of you.
//   view "head"     first person at the head D435i (default)
//   view "gripper"  first person at the gripper D405 (its own RGB-D surface)
//   view "overview" behind and above the robot
const VIEWS = ['head', 'gripper', 'overview'];
const VIEW_NAMES = { head: 'Head camera (RealSense D435i)', gripper: 'Gripper camera (RealSense D405)', overview: 'Overview (behind Stretch)' };
const euler = new THREE.Euler(0, 0, 0, 'YXZ');
const ref = { yaw: 0, pitch: 0, pos: new THREE.Vector3(0, 1.6, 0), set: false };
S.view = 'head';
function viewerPose() {
  const cam = renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;
  const p = new THREE.Vector3(), q = new THREE.Quaternion();
  cam.matrixWorld.decompose(p, q, new THREE.Vector3());
  euler.setFromQuaternion(q, 'YXZ');
  return { p, yaw: euler.y, pitch: euler.x };
}
const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));
function setReference() {
  const v = viewerPose();
  ref.yaw = v.yaw; ref.pitch = v.pitch; ref.pos.copy(v.p); ref.set = true;
  ui.rotation.y = ref.yaw;
  gripView.position.copy(ref.pos); gripView.rotation.set(0, ref.yaw, 0);
  placeStage(true);
  sendLook(true);
}
// the map point that should sit at your reference pose, and the map heading that should face ref.yaw
function anchor() {
  const r = S.robot, [x, y, th] = (r && r.base) || [0, 0, 0];
  const camZ = S.T ? S.T.elements[14] : 1.3;
  if (S.view === 'overview') return { x: x - 1.7 * Math.cos(th), y: y - 1.7 * Math.sin(th), z: 1.9, psi: th };
  return { x, y, z: camZ, psi: th };   // head (and gripper, for the hidden stage): at the head, facing the front
}
function placeStage(snapNow = false) {
  const a = anchor();
  const yaw = ref.yaw - a.psi;
  const p3 = new THREE.Vector3(a.x, a.y, a.z).applyMatrix4(new THREE.Matrix4().makeRotationY(yaw).multiply(M_MAP));
  const tx = ref.pos.x - p3.x, ty = ref.pos.y - p3.y, tz = ref.pos.z - p3.z;
  const k = snapNow ? 1 : 0.12;   // smooth the ride-along (SLAM pose updates, base motion)
  place.yaw += k * wrapPi(yaw - place.yaw);
  place.tx += k * (tx - place.tx); place.ty += k * (ty - place.ty); place.tz += k * (tz - place.tz);
  updateStage();
}
function setView(v) {
  S.view = v;
  const grip = v === 'gripper';
  aHead.visible = aMast.visible = v !== 'head';   // you ARE the head in this view
  stage.visible = !grip;
  gripView.visible = grip;
  gripPanel.mesh.visible = !grip && !!gripPanel.tex.image;
  placeStage(true);
  toast(VIEW_NAMES[v]);
}
function nextView() { setView(VIEWS[(VIEWS.indexOf(S.view) + 1) % VIEWS.length]); }
// gripper D405 as an immersive RGB-D surface locked to your reference frame (you "are" the gripper camera)
const gripView = new THREE.Group(); gripView.visible = false; scene.add(gripView);
let gripLive = null;

// ------------------------------------------------------------------ head look (telepresence, relative)
let lastLookSent = 0;
function sendLook(force = false) {
  const now = performance.now();
  if (!force && now - lastLookSent < 100) return;
  lastLookSent = now;
  const active = renderer.xr.isPresenting && S.headControl && ref.set && S.view !== 'gripper';
  const { yaw, pitch } = viewerPose();
  // 0, 0 = the robot head looks to the front; the robot adds your rotation since the reference
  send({ type: 'look', yaw: wrapPi(yaw - ref.yaw), pitch: pitch - ref.pitch, active, frame: 'relative' });
}

// ------------------------------------------------------------------ expert teleop
function claim() { enableMic(); send({ type: 'teleop_claim', pin: $('pin').value.trim() }); }
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
  $('talk').disabled = st !== 'granted';
}
function setTeleopOn(on) {
  S.teleopOn = on;
  send({ type: 'teleop_pause', paused: !on });
  toast(on ? 'Expert control ON - the routine runs in the background' : 'Control paused - Stretch resumes its routine');
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
let leftGripSince = 0, lastTeleopSent = 0, ySince = 0;
const tmpM = new THREE.Matrix4(), tmpP = new THREE.Vector3(), tmpQ = new THREE.Quaternion(), tmpS = new THREE.Vector3();
const M_MAP_INV = M_MAP.clone().invert();
// Hand pose in the ROBOT BASE frame (x forward, y left, z up): your reference forward = the robot's forward,
// so "hand to the right" is the robot's right no matter how the world rides along.
function toBase(objMatrixWorld) {
  tmpM.copy(M_MAP_INV).multiply(new THREE.Matrix4().makeRotationY(-ref.yaw))
    .multiply(new THREE.Matrix4().makeTranslation(-ref.pos.x, -ref.pos.y, -ref.pos.z)).multiply(objMatrixWorld);
  tmpM.decompose(tmpP, tmpQ, tmpS);
  return { p: [tmpP.x, tmpP.y, tmpP.z], q: [tmpQ.x, tmpQ.y, tmpQ.z, tmpQ.w] };
}
function jointDist(hand, a, b) {
  const ja = hand.joints[a], jb = hand.joints[b];
  if (!ja || !jb || !ja.visible || !jb.visible) return null;
  return ja.position.distanceTo(jb.position);
}
const clamp1 = (x) => Math.max(-1, Math.min(1, x));
function pollInputs(now) {
  const session = renderer.xr.getSession();
  if (!session) return;
  const cmd = { type: 'teleop', drive: [0, 0], clutch: false, precision: false, wrist: S.wrist, gripper: null,
    hand: null, mode: 'controllers', frame: 'base' };
  let haveRightCtl = false, ldrive = [0, 0], rdrive = [0, 0];
  for (const src of session.inputSources) {
    const side = src.handedness;
    if (side !== 'left' && side !== 'right') continue;
    inputs[side] = src;
    const gp = src.gamepad;
    if (src.hand || !gp) continue;
    const prev = prevBtn[side] || [];
    const b = (i) => gp.buttons[i] && gp.buttons[i].pressed;
    const edge = (i) => b(i) && !prev[i];
    const released = (i) => !b(i) && prev[i];
    if (side === 'left') {
      ldrive = [-(gp.axes[3] || 0), -(gp.axes[2] || 0)];                    // stick: drive
      if (edge(4)) { S.headControl = !S.headControl; sendLook(true); toast(`Head follows you: ${S.headControl ? 'ON' : 'off'}`); } // X
      if (edge(5)) ySince = now;                                            // Y: tap = next camera, hold = overlays
      if (b(5) && ySince && now - ySince > 900) { setOverlay(!S.overlay); ySince = 0; }
      if (released(5) && ySince) { nextView(); ySince = 0; }
      if (edge(0)) pttStart();                                              // trigger: push-to-talk
      if (released(0)) pttStop();
      if (edge(3) && S.teleop === 'granted') setTeleopOn(!S.teleopOn);     // stick click
      if (b(1)) { if (!leftGripSince) leftGripSince = now; if (now - leftGripSince > 800) { ackAlarm(); leftGripSince = now + 1e9; } }
      else leftGripSince = 0;
    } else {
      haveRightCtl = true;
      rdrive = [-(gp.axes[3] || 0), -(gp.axes[2] || 0)];                    // stick: fine drive (40 %)
      cmd.clutch = b(1);
      cmd.precision = gp.buttons[0] && gp.buttons[0].value > 0.5;
      if (edge(4)) cmd.gripper = 'open';                                    // A
      if (edge(5)) cmd.gripper = 'close';                                   // B
      if (edge(3)) { S.wrist = !S.wrist; cmd.wrist = S.wrist; }            // stick click
      const gi = sideIndex('right');
      if (gi >= 0) cmd.hand = toBase(grips[gi].matrixWorld);
    }
    prevBtn[side] = gp.buttons.map((x) => x.pressed);
  }
  cmd.drive = [clamp1(ldrive[0] + 0.4 * rdrive[0]), clamp1(ldrive[1] + 0.4 * rdrive[1])];
  if (!haveRightCtl) {   // bare hands: left pinch = clutch, right wrist = arm, right thumb-index = gripper
    const ri = sideIndex('right'), li = sideIndex('left');
    if (ri >= 0 && handOf[ri] && handOf[ri].hand) {
      const rh = hands[ri], wrist = rh.joints.wrist;
      if (wrist && wrist.visible) {
        cmd.mode = 'hands';
        cmd.hand = toBase(wrist.matrixWorld);
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

// ------------------------------------------------------------------ push-to-talk (Quest microphone -> Stretch)
const mic = { stream: null, ctx: null, node: null, on: false, rate: 48000 };
const WORKLET = `class Tap extends AudioWorkletProcessor { process(i) { if (i[0] && i[0][0]) this.port.postMessage(i[0][0].slice(0)); return true; } }
registerProcessor('tap', Tap);`;
async function enableMic() {
  if (mic.stream) return true;
  try {
    mic.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
    mic.ctx = new AudioContext();
    mic.rate = mic.ctx.sampleRate;
    await mic.ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' })));
    const src = mic.ctx.createMediaStreamSource(mic.stream);
    mic.node = new AudioWorkletNode(mic.ctx, 'tap');
    let acc = [];
    mic.node.port.onmessage = (e) => {
      if (!mic.on) return;
      acc.push(e.data);
      if (acc.reduce((n, a) => n + a.length, 0) >= mic.rate / 10) { sendAudio(acc); acc = []; }   // ~100 ms chunks
    };
    src.connect(mic.node);
    $('mic').textContent = 'Mic ready'; $('mic').disabled = true;
    return true;
  } catch (e) {
    toast(`Microphone unavailable: ${e.message || e}`);
    return false;
  }
}
function sendAudio(chunks) {   // resample to 16 kHz mono PCM16; message = [u8 10][pcm16 little-endian]
  const n = chunks.reduce((k, a) => k + a.length, 0), all = new Float32Array(n);
  let o = 0; for (const a of chunks) { all.set(a, o); o += a.length; }
  const ratio = mic.rate / 16000, m = Math.floor((n - 1) / ratio);
  const pkt = new Uint8Array(1 + 2 * m), dv = new DataView(pkt.buffer);
  pkt[0] = 10;
  for (let i = 0; i < m; i++) {
    const x = i * ratio, j = Math.floor(x), f = x - j;
    const v = all[j] * (1 - f) + all[j + 1] * f;
    dv.setInt16(1 + 2 * i, Math.max(-32768, Math.min(32767, Math.round(v * 32767))), true);
  }
  if (ws && ws.readyState === 1) ws.send(pkt);
}
async function pttStart() {
  if (S.teleop !== 'granted') { toast('Take control (PIN) to talk through Stretch'); return; }
  if (!(await enableMic())) return;
  if (mic.ctx.state === 'suspended') await mic.ctx.resume();
  mic.on = true; S.voice = { state: 'listening' };
  send({ type: 'ptt', state: 'start' });
}
function pttStop() {
  if (!mic.on) return;
  mic.on = false;
  send({ type: 'ptt', state: 'stop' });
}
$('mic').onclick = enableMic;
$('talk').addEventListener('pointerdown', pttStart);
['pointerup', 'pointerleave'].forEach((ev) => $('talk').addEventListener(ev, pttStop));

// ------------------------------------------------------------------ control-barrier lines (lidar obstacles)
const barrierGroup = new THREE.Group(); stage.add(barrierGroup);
const BARRIER_COLORS = { active: 0xd03b3b, near: 0xfab219, clear: 0x3987e5 };
function applyBarriers(msg) {
  for (const c of [...barrierGroup.children]) { barrierGroup.remove(c); c.geometry.dispose(); c.material.dispose(); }
  for (const ln of msg.lines || []) {
    const col = BARRIER_COLORS[ln.level] || 0x3987e5;
    const op = ln.level === 'clear' ? 0.45 : 0.95;
    for (const z of [0.03, 0.35]) {   // a low "fence": the keep-out line on the floor and at shin height
      const pts = ln.pts.map((p) => new THREE.Vector3(p[0], p[1], z));
      const Kind = ln.closed ? THREE.LineLoop : THREE.Line;
      barrierGroup.add(new Kind(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: op })));
    }
    const posts = [];
    for (let i = 0; i < ln.pts.length; i += Math.max(1, Math.floor(ln.pts.length / 24))) {
      posts.push(new THREE.Vector3(ln.pts[i][0], ln.pts[i][1], 0.03), new THREE.Vector3(ln.pts[i][0], ln.pts[i][1], 0.35));
    }
    barrierGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(posts),
      new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: op * 0.6 })));
  }
  if (msg.d && !footCircles.userData.d) {   // the footprint circles the barriers protect
    footCircles.userData.d = msg.d;
    for (const lx of [0.08, -0.08]) {
      const pts = []; for (let i = 0; i <= 48; i++) { const a = i / 48 * 2 * Math.PI; pts.push(new THREE.Vector3(lx + 0.21 * Math.cos(a), 0.21 * Math.sin(a), 0.03)); }
      footCircles.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x3987e5 })));
    }
  }
  S.barriers = (msg.lines || []).length;
  S.barrierActive = (msg.lines || []).some((l) => l.level === 'active');
}
const footCircles = new THREE.Group(); avatar.add(footCircles);

// ------------------------------------------------------------------ scene graph (labelled furniture boxes)
const graphGroup = new THREE.Group(); stage.add(graphGroup);
const GRAPH_COLORS = { table: 0x3987e5, chair: 0xe66767, sofa: 0x9b6bd6, bed: 0x9b6bd6, bench: 0xe66767,
  cabinet: 0x19a979, screen: 0xfab219, laptop: 0xfab219, low_object: 0x8a8a8a };
const graphLabels = new Map();
function graphLabel(text, color) {
  const cv = document.createElement('canvas'); cv.width = 256; cv.height = 64;
  const x = cv.getContext('2d');
  x.fillStyle = 'rgba(20,20,22,0.78)'; x.beginPath(); x.roundRect(0, 8, 256, 48, 12); x.fill();
  x.fillStyle = '#' + color.toString(16).padStart(6, '0'); x.fillRect(12, 24, 16, 16);
  x.fillStyle = '#eee'; x.font = '28px system-ui'; x.fillText(text, 38, 42);
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.scale.set(0.36, 0.09, 1); sp.renderOrder = 25; return sp;
}
function applySceneGraph(g) {
  S.scene = g.person || ''; S.sceneSupport = g.support; S.sceneAt = g.at;
  if (g.version === graphGroup.userData.version) return highlightGraph();
  graphGroup.userData.version = g.version;
  for (const c of [...graphGroup.children]) { graphGroup.remove(c); c.geometry?.dispose(); c.material?.map?.dispose(); c.material?.dispose(); }
  graphLabels.clear();
  for (const b of g.boxes || []) {
    if (b.label === 'object' || b.label === 'tall_object') continue;
    const col = GRAPH_COLORS[b.label] || 0x8a8a8a;
    const [l, w, h] = b.s, c = Math.cos(b.yaw), s = Math.sin(b.yaw);
    const corner = (u, v, z) => new THREE.Vector3(b.c[0] + c * u - s * v, b.c[1] + s * u + c * v, z);
    const P = [[l, w], [-l, w], [-l, -w], [l, -w]].map(([u, v]) => [u / 2, v / 2]);
    const seg = [];
    for (let i = 0; i < 4; i++) {
      const [u0, v0] = P[i], [u1, v1] = P[(i + 1) % 4];
      seg.push(corner(u0, v0, 0.02), corner(u1, v1, 0.02), corner(u0, v0, h), corner(u1, v1, h), corner(u0, v0, 0.02), corner(u0, v0, h));
    }
    const box = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg),
      new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.55 }));
    box.userData.id = b.id; graphGroup.add(box);
    const sp = graphLabel(b.id.replace('_', ' '), col); sp.position.set(b.c[0], b.c[1], h + 0.12);
    graphGroup.add(sp); graphLabels.set(b.id, { box, sp });
  }
  highlightGraph();
}
function highlightGraph() {   // the object the person sits on / works at glows
  for (const [id, o] of graphLabels) {
    const hot = id === S.sceneSupport || id === S.sceneAt;
    o.box.material.opacity = hot ? 1.0 : 0.45; o.box.material.linewidth = hot ? 2 : 1;
  }
}

// ------------------------------------------------------------------ toasts (2-D page + HUD)
function toast(text) { S.toast = { text, t: performance.now() }; msg(text); }

// ------------------------------------------------------------------ overlays / toggles
function setOverlay(on) { S.overlay = on; heatGroup.visible = captionGroup.visible = barrierGroup.visible = graphGroup.visible = on; $('overlay').checked = on; }
$('overlay').onchange = (e) => setOverlay(e.target.checked);
$('ceil').onchange = (e) => { mapMat.uniforms.ceilingOn.value = e.target.checked ? 1 : 0; };
$('view').onclick = nextView;

// desktop / 2-D preview: drag to look, WASD to walk, V = next camera, O = bird's-eye
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
let birdsEye = false;
function overview() {   // 2-D preview only: bird's-eye view of the robot and the room (stops the ride-along)
  birdsEye = true;
  const r = S.robot, [x, y] = (r && r.base) || [0, 0];
  const c = new THREE.Vector3(x, y, 0).applyMatrix4(stage.matrix);
  camera.position.set(c.x + 3.2, 4.2, c.z + 3.2);
  camera.lookAt(c.x, 0.4, c.z);
}
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase(); keys.add(k);
  if (k === 'v') nextView();
  if (k === 'o') overview();
  if (k === 'c') { birdsEye = false; camera.position.copy(ref.pos); camera.quaternion.identity(); setReference(); }
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
  if (ref.set && !birdsEye) placeStage();   // ride along with the robot
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
    ref.set = false;
    session.addEventListener('end', () => { btn.textContent = 'Enter VR'; ref.set = false; sendLook(true); setTimeout(setReference, 100); });
    // your pose a moment after entering = the robot's "0 rotation" (head looking to the front)
    setTimeout(() => { setReference(); setView('head'); toast('Looking to the front: turn your head, Stretch follows'); }, 400);
  };
}
setOverlay(true);
setupVR();
connect();
window.__vr = { S, stage, overview, camera, applyMapDelta, slotOf, setView, nextView, setReference, ref };   // debugging

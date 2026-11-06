// 完全ブロッホ球 FFN を持つ文字単位 GPT の推論エンジン（scale_exp.GPT(kind='complete') と同じ計算）。
// K/V キャッシュで1文字ずつ計算し、文脈が ctx に達したら後半 ctx/2 文字を残して読み直す。
function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * f * 5.960464477539063e-8
      : e === 31 ? (f ? NaN : s * Infinity)
      : s * (1 + f / 1024) * Math.pow(2, e - 15);
  }
  return out;
}

function decodeWeights(b64, layout) {
  const bin = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const all = f16ToF32(new Uint16Array(bytes.buffer));
  const w = {};
  for (const [name, [off, shape]] of Object.entries(layout)) {
    w[name] = all.subarray(off, off + shape.reduce((a, b) => a * b, 1));
  }
  return w;
}

function decodeWeightsBuffers(buffers, layout) {
  // 分割した fp16 のバイナリ（weights_0.bin, weights_1.bin, …）をつなげて読む
  const total = buffers.reduce((a, b) => a + b.byteLength, 0), bytes = new Uint8Array(total);
  let off = 0; for (const b of buffers) { bytes.set(new Uint8Array(b), off); off += b.byteLength; }
  const all = f16ToF32(new Uint16Array(bytes.buffer));
  const w = {};
  for (const [name, [o, shape]] of Object.entries(layout)) w[name] = all.subarray(o, o + shape.reduce((a, b) => a * b, 1));
  return w;
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function linear(W, b, x, nOut, nIn, out) {
  out = out || new Float32Array(nOut);
  for (let o = 0; o < nOut; o++) {
    let s = b ? b[o] : 0; const row = o * nIn;
    for (let i = 0; i < nIn; i++) s += W[row + i] * x[i];
    out[o] = s;
  }
  return out;
}

function layerNorm(x, g, b) {
  const n = x.length; let mean = 0, v = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  const inv = 1 / Math.sqrt(v / n + 1e-5), out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * g[i] + b[i];
  return out;
}

class CompleteQBNN {
  constructor(meta, weights) {
    this.cfg = meta.config;
    this.chars = meta.chars;
    this.index = new Map(this.chars.map((c, i) => [c, i]));
    this.w = weights;
    this.special = meta.special ? Object.fromEntries(meta.special.map(t => [t.slice(1, -1), this.index.get(t)])) : null;
    this.D = this.cfg.d; this.L = this.cfg.layers; this.Hd = this.cfg.heads;
    this.H = this.cfg.h; this.N = this.H / 3; this.V = this.chars.length; this.ctx = this.cfg.ctx;
  }
  // 語彙にない文字は学習時と同じく <unk>（0番）にする
  encode(text) { return [...text].map(ch => this.index.has(ch) ? this.index.get(ch) : 0); }
  // 指示モード: <bos>指示<sep> の後に応答を書かせ、<eos> で止める
  encodeInstruction(text) { return [this.special.bos, ...this.encode(text), this.special.sep]; }
  initState() { return { tokens: [], K: Array.from({ length: this.L }, () => []), V: Array.from({ length: this.L }, () => []) }; }

  step(token, st, opts = {}) {
    if (st.tokens.length >= this.ctx) {                 // 文脈が一杯: 後半を残して読み直す
      const keep = st.tokens.slice(-this.ctx / 2);
      const fresh = this.initState();
      for (const t of keep) this._forward(t, fresh, {});
      Object.assign(st, fresh);
    }
    return this._forward(token, st, opts);
  }

  _forward(token, st, opts) {
    const { measure = 0, rng = Math.random } = opts;
    const { D, Hd, H, N } = this, hd = D / Hd, pos = st.tokens.length, w = this.w;
    let x = new Float32Array(D);
    for (let i = 0; i < D; i++) x[i] = w.tok[token * D + i] + w.pos[pos * D + i];
    const bloch = [], trace = [];
    for (let l = 0; l < this.L; l++) {
      const p = 'b' + l + '.';
      const qkv = linear(w[p + 'qkv.w'], w[p + 'qkv.b'], layerNorm(x, w[p + 'ln1.w'], w[p + 'ln1.b']), 3 * D, D);
      const q = qkv.subarray(0, D);
      st.K[l].push(qkv.slice(D, 2 * D)); st.V[l].push(qkv.slice(2 * D, 3 * D));
      const y = new Float32Array(D), T = st.K[l].length, sc = new Float32Array(T), scale = 1 / Math.sqrt(hd);
      for (let h = 0; h < Hd; h++) {
        const o = h * hd; let mx = -Infinity;
        for (let s = 0; s < T; s++) { const k = st.K[l][s]; let d = 0; for (let i = 0; i < hd; i++) d += q[o + i] * k[o + i]; sc[s] = d * scale; if (sc[s] > mx) mx = sc[s]; }
        let z = 0; for (let s = 0; s < T; s++) { sc[s] = Math.exp(sc[s] - mx); z += sc[s]; }
        for (let s = 0; s < T; s++) { const v = st.V[l][s], a = sc[s] / z; for (let i = 0; i < hd; i++) y[o + i] += a * v[o + i]; }
      }
      const att = linear(w[p + 'proj.w'], w[p + 'proj.b'], y, D, D);
      for (let i = 0; i < D; i++) x[i] += att[i];
      // 完全ブロッホ球 FFN: up -> (c, u, v) -> b = R m P(q) -> down
      const a = linear(w[p + 'up.w'], w[p + 'up.b'], layerNorm(x, w[p + 'ln2.w'], w[p + 'ln2.b']), H, D);
      const R = w[p + 'R'], b = new Float32Array(H);
      const info = { c: new Float32Array(N), u: new Float32Array(N), v: new Float32Array(N), measured: new Uint8Array(N) };
      for (let j = 0; j < N; j++) {
        const c = a[3 * j], u = a[3 * j + 1], v = a[3 * j + 2];
        const m = Math.tanh(c), inv = 1 / Math.sqrt(1 + u * u + v * v);
        const q0 = m * u * inv, q1 = m * v * inv, q2 = m * inv, r = R.subarray(9 * j, 9 * j + 9);
        let bx = r[0] * q0 + r[1] * q1 + r[2] * q2, by = r[3] * q0 + r[4] * q1 + r[5] * q2, bz = r[6] * q0 + r[7] * q1 + r[8] * q2;
        if (measure > 0 && rng() < measure) {
          const n = Math.hypot(bx, by, bz), ax = n > 1e-12 ? [bx / n, by / n, bz / n] : [0, 0, 1];
          const sgn = rng() < (1 + n) / 2 ? 1 : -1;
          bx = sgn * ax[0]; by = sgn * ax[1]; bz = sgn * ax[2]; info.measured[j] = 1;
        }
        b[3 * j] = bx; b[3 * j + 1] = by; b[3 * j + 2] = bz;
        info.c[j] = c; info.u[j] = u; info.v[j] = v;
      }
      const f = linear(w[p + 'down.w'], w[p + 'down.b'], b, D, H);
      for (let i = 0; i < D; i++) x[i] += f[i];
      bloch.push(b); trace.push(info);
    }
    st.tokens.push(token);
    const xf = layerNorm(x, w['ln.w'], w['ln.b']);
    const logits = linear(w.tok, null, xf, this.V, D);
    return { logits, state: st, bloch, trace };
  }

  static softmax(logits, temperature) {
    const t = temperature > 0 ? temperature : 1;
    let mx = -Infinity; for (const v of logits) mx = Math.max(mx, v);
    const p = new Float32Array(logits.length); let z = 0;
    for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i] - mx) / t); z += p[i]; }
    for (let i = 0; i < p.length; i++) p[i] /= z;
    return p;
  }

  static sample(p, rng) {
    let r = rng(), acc = 0;
    for (let i = 0; i < p.length; i++) { acc += p[i]; if (r < acc) return i; }
    return p.length - 1;
  }
}

if (typeof module !== 'undefined') module.exports = { CompleteQBNN, decodeWeights, decodeWeightsBuffers, mulberry32 };

// 完全QBNN 文字言語モデルの推論エンジン（complete_qbnn.CompleteQBNNLM と同じ計算）
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
    const size = shape.reduce((a, b) => a * b, 1);
    w[name] = all.subarray(off, off + size);
  }
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

class CompleteQBNN {
  constructor(meta, weights) {
    this.cfg = meta.config;
    this.chars = meta.chars;
    this.index = new Map(this.chars.map((c, i) => [c, i]));
    this.w = weights;
    this.N = this.cfg.neurons;
    this.E = this.cfg.embed_dim;
    this.L = this.cfg.layers;
    this.V = this.cfg.vocab_size;
  }
  encode(text) { const ids = []; for (const ch of text) if (this.index.has(ch)) ids.push(this.index.get(ch)); return ids; }
  initState() { return Array.from({ length: this.L }, () => new Float32Array(3 * this.N)); }

  // 1ステップ: token -> logits と新しい状態。trace には各層のニューロンの (c, u, v, b) を入れる
  step(token, state, opts = {}) {
    const { measure = 0, rng = Math.random } = opts;
    const N = this.N, E = this.E;
    let x = this.w.embed.subarray(token * E, token * E + E);
    const newState = [], trace = [];
    for (let l = 0; l < this.L; l++) {
      const W = this.w['W' + l], B = this.w['B' + l], R = this.w['R' + l];
      const inp = new Float32Array(x.length + 3 * N);
      inp.set(x, 0); inp.set(state[l], x.length);
      const I = inp.length, pre = new Float32Array(3 * N);
      for (let o = 0; o < 3 * N; o++) {
        let s = B[o]; const row = o * I;
        for (let i = 0; i < I; i++) s += W[row + i] * inp[i];
        pre[o] = s;
      }
      const b = new Float32Array(3 * N), info = { c: new Float32Array(N), u: new Float32Array(N), v: new Float32Array(N), measured: new Uint8Array(N) };
      for (let j = 0; j < N; j++) {
        const c = pre[3 * j], u = pre[3 * j + 1], v = pre[3 * j + 2];
        const m = Math.tanh(c), inv = 1 / Math.sqrt(1 + u * u + v * v);
        const q0 = m * u * inv, q1 = m * v * inv, q2 = m * inv;             // CB4: b = m P(q)
        const r = R.subarray(9 * j, 9 * j + 9);                              // 1量子ビットゲート
        let bx = r[0] * q0 + r[1] * q1 + r[2] * q2;
        let by = r[3] * q0 + r[4] * q1 + r[5] * q2;
        let bz = r[6] * q0 + r[7] * q1 + r[8] * q2;
        if (measure > 0 && rng() < measure) {                                // CB5: 葉軸で測定
          const n = Math.hypot(bx, by, bz);
          const ax = n > 1e-12 ? [bx / n, by / n, bz / n] : [0, 0, 1];
          const sgn = rng() < (1 + n) / 2 ? 1 : -1;
          bx = sgn * ax[0]; by = sgn * ax[1]; bz = sgn * ax[2];
          info.measured[j] = 1;
        }
        b[3 * j] = bx; b[3 * j + 1] = by; b[3 * j + 2] = bz;
        info.c[j] = c; info.u[j] = u; info.v[j] = v;
      }
      newState.push(b); trace.push(info);
      x = b;
    }
    const Wo = this.w.Wo, Bo = this.w.Bo, H = 3 * N, logits = new Float32Array(this.V);
    for (let k = 0; k < this.V; k++) {
      let s = Bo[k]; const row = k * H;
      for (let i = 0; i < H; i++) s += Wo[row + i] * x[i];
      logits[k] = s;
    }
    return { logits, state: newState, trace };
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

if (typeof module !== 'undefined') module.exports = { CompleteQBNN, decodeWeights, mulberry32 };

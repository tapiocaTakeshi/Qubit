const fs = require('fs');
const { CompleteQBNN, decodeWeights } = require('./engine_gpt.js');
const meta = JSON.parse(fs.readFileSync('meta_gpt.json', 'utf8'));
const ref = JSON.parse(fs.readFileSync('ref_gpt.json', 'utf8'));
const m = new CompleteQBNN(meta, decodeWeights(fs.readFileSync('weights_gpt.b64', 'utf8'), meta.layout));
const ids = m.special ? m.encodeInstruction(ref.prompt) : m.encode(ref.prompt);
let st = m.initState(), out;
for (const t of ids) out = m.step(t, st);
let d = 0; out.logits.forEach((v, i) => d = Math.max(d, Math.abs(v - ref.logits_first[i])));
console.log('max |logit JS - torch| =', d.toExponential(2));
st = m.initState(); for (const t of ids.slice(0, -1)) m.step(t, st);
let tok = ids[ids.length - 1], text = '', t0 = Date.now();
for (let i = 0; i < 200; i++) { const o = m.step(tok, st); tok = o.logits.indexOf(Math.max(...o.logits)); text += m.chars[tok]; }
let same = 0; while (same < text.length && text[same] === ref.greedy[same]) same++;
console.log(`greedy 200 chars: identical prefix ${same}/200, ${((Date.now() - t0) / 200).toFixed(1)} ms/char`);
console.log('JS   :', text.slice(0, 80).replace(/\n/g, '⏎')); console.log('torch:', ref.greedy.slice(0, 80).replace(/\n/g, '⏎'));

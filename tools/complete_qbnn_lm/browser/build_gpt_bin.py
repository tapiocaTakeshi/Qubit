import json, os, sys
d, train, params = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(f'{d}/template_chat_bin.html', encoding='utf8').read()
meta = json.load(open(f'{d}/meta_gpt.json', encoding='utf8')); c = meta['config']
parts = sorted(f for f in os.listdir(d) if f.startswith('weights_') and f.endswith(('.bin','.wasm')))
meta['parts'] = len(parts); meta['megabytes'] = round(sum(os.path.getsize(f'{d}/{f}') for f in parts) / 1e6)
engine = open(f'{d}/engine_gpt.js', encoding='utf8').read().replace("if (typeof module !== 'undefined') module.exports = { CompleteQBNN, decodeWeights, decodeWeightsBuffers, mulberry32 };", '')
out = (t.replace('%%LAYERS%%', str(c['layers'])).replace('%%NEURONS%%', str(c['h'] // 3)).replace('%%CTX%%', str(c['ctx']))
        .replace('%%TRAIN%%', train).replace('%%PARAMS%%', params)
        .replace('%%META%%', json.dumps(meta, ensure_ascii=False).replace('</', '<\\/')).replace('%%ENGINE%%', engine))
open(f'{d}/observatory_L.html', 'w', encoding='utf8').write(out)
print('page bytes', len(out.encode()), 'parts', parts, 'leftover', [p for p in ('%%LAYERS%%', '%%META%%', '%%ENGINE%%', '%%TRAIN%%') if p in out])

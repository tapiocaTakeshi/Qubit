#!/usr/bin/env python3
"""完全QBNN 観測室（ブラウザで動く CompleteQBNNLM の可視化ページ）を組み立てる.

    python tools/complete_qbnn_viewer/build.py complete_qbnn_lm.pt -o complete-qbnn-observatory.html

学習済みの CompleteQBNNAI チェックポイントの重みを fp16 にして base64 でページに埋め込む.
ゲートは前もって SO(3) 行列にしておき, engine.js が complete_qbnn.CompleteQBNNLM と
同じ順伝播をブラウザで計算する.
"""

import argparse
import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import complete_qbnn as cq  # noqa: E402


def export(ckpt):
    ai = cq.CompleteQBNNAI.load(ckpt)
    cfg, sd = ai.model.config, {k: v.detach() for k, v in ai.model.state_dict().items()}
    parts = [('embed', sd['embed.weight'])]
    for i in range(cfg['layers']):
        parts += [(f'W{i}', sd[f'cells.{i}.proj.weight']), (f'B{i}', sd[f'cells.{i}.proj.bias']),
                  (f'R{i}', cq.axis_angle_rotation(sd[f'cells.{i}.gate'].double()))]
    parts += [('Wo', sd['readout.weight']), ('Bo', sd['readout.bias'])]
    blobs, layout, off = [], {}, 0
    for name, t in parts:
        a = t.float().numpy().astype(np.float16).ravel()
        layout[name] = [off, list(t.shape)]
        off += a.size
        blobs.append(a)
    meta = {'config': cfg, 'chars': ai.chars, 'layout': layout}
    return meta, base64.b64encode(np.concatenate(blobs).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('-o', '--out', default='complete-qbnn-observatory.html')
    args = ap.parse_args()
    meta, weights = export(args.ckpt)
    template = open(os.path.join(HERE, 'template.html'), encoding='utf-8').read()
    engine = open(os.path.join(HERE, 'engine.js'), encoding='utf-8').read()
    engine = engine.replace("if (typeof module !== 'undefined') module.exports = { CompleteQBNN, decodeWeights, mulberry32 };", '')
    page = (template.replace('%%META%%', json.dumps(meta, ensure_ascii=False).replace('</', '<\\/'))
            .replace('%%WEIGHTS%%', weights).replace('%%ENGINE%%', engine))
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(page)
    print(f"wrote {args.out} ({len(page.encode()) / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()

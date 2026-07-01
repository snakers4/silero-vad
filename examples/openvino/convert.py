#!/usr/bin/env python3
"""Convert the Silero VAD ONNX model into an OpenVINO compatible one.

The stock silero_vad.onnx does not load in OpenVINO: the ONNX frontend
cannot infer static ranks for the Conv ops (the STFT) inside the If
subgraphs. The graph contains one top level If that switches between the
8 kHz and 16 kHz paths, plus 7 nested If nodes per branch left behind by
TorchScript tracing. None of their conditions depend on the audio itself,
only on the sample rate and tensor shapes, so for a fixed configuration
they can all be resolved offline.

This script inlines every If by evaluating its condition empirically with
onnxruntime, removes dead code (including the weights of the unused sample
rate branch), folds Identity nodes (recent OpenVINO serializes them as
opset16::Identity, which older CPU plugins cannot execute) and fixes
static shapes. The result loads with core.read_model() and compiles on
CPU directly, both as ONNX and as exported IR.

Usage:
    python convert.py [--sr 16000] [--save-ir] [input.onnx] [output.onnx]

Defaults: input is the model shipped in this repo, output is
silero_vad_16k.onnx or silero_vad_8k.onnx depending on --sr.

Requires: numpy, onnx, onnxruntime. OpenVINO only if --save-ir is used.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, shape_inference

REPO_MODEL = Path(__file__).resolve().parents[2] / \
    'src' / 'silero_vad' / 'data' / 'silero_vad.onnx'


def defined_names(graph):
    """Names defined inside a graph (ONNX scoping rules)."""
    s = set(i.name for i in graph.initializer)
    s |= set(i.name for i in graph.input)
    for n in graph.node:
        s |= set(o for o in n.output if o)
    return s


def deep_rename(node, rename):
    """Rename references in a node, recursing into subgraphs with shadowing."""
    for i, name in enumerate(node.input):
        if name in rename:
            node.input[i] = rename[name]
    for i, name in enumerate(node.output):
        if name in rename:
            node.output[i] = rename[name]
    for attr in node.attribute:
        subgraphs = []
        if attr.type == onnx.AttributeProto.GRAPH:
            subgraphs = [attr.g]
        elif attr.type == onnx.AttributeProto.GRAPHS:
            subgraphs = list(attr.graphs)
        for sg in subgraphs:
            shadowed = defined_names(sg)
            inner = {k: v for k, v in rename.items() if k not in shadowed}
            if inner:
                for sn in sg.node:
                    deep_rename(sn, inner)
                for o in sg.output:
                    if o.name in inner:
                        o.name = inner[o.name]


def eval_cond(model, cond_name, feeds):
    """Evaluate an If condition tensor by running the current model."""
    probe = copy.deepcopy(model)
    vi = helper.make_tensor_value_info(cond_name, TensorProto.BOOL, None)
    probe.graph.output.append(vi)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(probe.SerializeToString(), so,
                                providers=['CPUExecutionProvider'])
    out_names = [o.name for o in sess.get_outputs()]
    needed = {i.name: feeds[i.name] for i in sess.get_inputs()}
    res = sess.run(out_names, needed)
    val = np.asarray(res[out_names.index(cond_name)]).reshape(-1)
    assert val.size == 1, f"condition {cond_name} is not a scalar"
    return bool(val[0])


def inline_if(graph, if_idx, branch, prefix):
    """Replace graph.node[if_idx] (an If) with the nodes of `branch`."""
    if_node = graph.node[if_idx]
    internal = set(i.name for i in branch.initializer)
    for n in branch.node:
        internal |= set(o for o in n.output if o)

    rename = {name: f"{prefix}::{name}" for name in internal}

    extra_identity = []
    for b_out, if_out in zip(branch.output, if_node.output):
        if b_out.name in internal:
            rename[b_out.name] = if_out
        else:
            # branch output is a tensor captured from the outer scope
            extra_identity.append(helper.make_node(
                'Identity', [b_out.name], [if_out],
                name=f"{prefix}::identity_{if_out}"))

    new_nodes = []
    for n in branch.node:
        n2 = copy.deepcopy(n)
        n2.name = f"{prefix}::{n2.name}" if n2.name else f"{prefix}::anon"
        deep_rename(n2, rename)
        new_nodes.append(n2)
    new_nodes.extend(extra_identity)

    for init in branch.initializer:
        init2 = copy.deepcopy(init)
        init2.name = rename.get(init.name, init.name)
        graph.initializer.append(init2)

    nodes = list(graph.node)
    nodes[if_idx:if_idx + 1] = new_nodes
    del graph.node[:]
    graph.node.extend(nodes)


def eliminate_identity(graph):
    """Fold Identity nodes. They are pure metadata, but older OpenVINO CPU
    plugins cannot execute opset16::Identity in serialized IR files."""
    changed = True
    while changed:
        changed = False
        out_names = set(o.name for o in graph.output)
        for idx, n in enumerate(graph.node):
            if n.op_type != 'Identity':
                continue
            src, dst = n.input[0], n.output[0]
            if dst in out_names:
                if src in out_names:
                    continue  # output to output aliasing, keep it
                producer = next((p for p in graph.node if src in p.output), None)
                if producer is None:
                    continue  # src is a graph input or initializer, keep it
                for i, o in enumerate(producer.output):
                    if o == src:
                        producer.output[i] = dst
                for p in graph.node:
                    if p is not n:
                        for i, inp in enumerate(p.input):
                            if inp == src:
                                p.input[i] = dst
            else:
                for p in graph.node:
                    for i, inp in enumerate(p.input):
                        if inp == dst:
                            p.input[i] = src
            del graph.node[idx]
            changed = True
            break


def dce(graph):
    """Drop nodes, initializers and inputs not reachable from the outputs."""
    needed = set(o.name for o in graph.output)
    kept = []
    for n in reversed(list(graph.node)):
        if any(o in needed for o in n.output):
            kept.append(n)
            needed |= set(i for i in n.input if i)
            for attr in n.attribute:
                sgs = [attr.g] if attr.type == onnx.AttributeProto.GRAPH \
                    else list(attr.graphs) if attr.type == onnx.AttributeProto.GRAPHS else []
                for sg in sgs:
                    local = defined_names(sg)
                    for sn in sg.node:
                        needed |= set(i for i in sn.input if i and i not in local)
    kept.reverse()
    del graph.node[:]
    graph.node.extend(kept)

    keep_init = [i for i in graph.initializer if i.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(keep_init)
    keep_in = [i for i in graph.input if i.name in needed]
    del graph.input[:]
    graph.input.extend(keep_in)


def convert(src, dst, sr, save_ir=False):
    samples = 576 if sr == 16000 else 288  # new samples plus context
    model = onnx.load(str(src))
    feeds = {
        'input': np.zeros((1, samples), dtype=np.float32),
        'state': np.zeros((2, 1, 128), dtype=np.float32),
        'sr':    np.array(sr, dtype=np.int64),
    }

    step = 0
    while True:
        ifs = [(i, n) for i, n in enumerate(model.graph.node) if n.op_type == 'If']
        if not ifs:
            break
        idx, node = ifs[0]
        cond = eval_cond(model, node.input[0], feeds)
        branch_name = 'then_branch' if cond else 'else_branch'
        branch = next(a.g for a in node.attribute if a.name == branch_name)
        print(f"[{step}] inline If '{node.name}': cond={cond} -> {branch_name} "
              f"({len(branch.node)} nodes)")
        inline_if(model.graph, idx, branch, f"F{step}")
        step += 1

    assert not any(n.op_type in ('If', 'Loop', 'Scan') for n in model.graph.node)

    dce(model.graph)
    eliminate_identity(model.graph)

    for i in model.graph.input:
        dims = {'input': [1, samples], 'state': [2, 1, 128]}[i.name]
        for d, v in zip(i.type.tensor_type.shape.dim, dims):
            d.ClearField('dim_param')
            d.dim_value = v
    for o in model.graph.output:
        dims = {'output': [1, 1], 'stateN': [2, 1, 128]}[o.name]
        for d, v in zip(o.type.tensor_type.shape.dim, dims):
            d.ClearField('dim_param')
            d.dim_value = v
    del model.graph.value_info[:]

    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))

    n_if = sum(1 for n in model.graph.node if n.op_type == 'If')
    n_id = sum(1 for n in model.graph.node if n.op_type == 'Identity')
    print(f"\nwrote {dst}")
    print(f"  nodes: {len(model.graph.node)}, If left: {n_if}, "
          f"Identity left: {n_id}, inputs: {[i.name for i in model.graph.input]}")

    if save_ir:
        import openvino as ov
        ir_path = Path(dst).with_suffix('.xml')
        ov.save_model(ov.Core().read_model(str(dst)), str(ir_path),
                      compress_to_fp16=False)
        print(f"  IR written to {ir_path} (regenerate with your target "
              f"OpenVINO version when deploying)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('input', nargs='?', default=str(REPO_MODEL),
                   help='path to the stock silero_vad.onnx')
    p.add_argument('output', nargs='?', default=None,
                   help='output path (default: silero_vad_16k.onnx or _8k)')
    p.add_argument('--sr', type=int, choices=[8000, 16000], default=16000,
                   help='sample rate branch to keep (default 16000)')
    p.add_argument('--save-ir', action='store_true',
                   help='also export OpenVINO IR next to the output')
    a = p.parse_args()
    out = a.output or f"silero_vad_{a.sr // 1000}k.onnx"
    convert(a.input, out, a.sr, a.save_ir)


if __name__ == '__main__':
    main()

"""Lossless VRAM placement for the LongCat-Video-Avatar DiT on a single 32 GB GPU.

Every transformer block keeps a master copy of its weights in pinned CPU RAM.
  * resident blocks: a GPU copy is kept between uses (dropped while "parked", e.g. during VAE decode)
  * streamed blocks: copied into one of two reusable GPU slots right before they run
    (double-buffered on a side CUDA stream, so the copy of the next block overlaps compute).

The math never changes: the very same weight bits are fed to the very same kernels in the very same
order, so outputs are bit-identical to having the whole model on the GPU (verified by test_lossless.py).
"""
import torch


def _pin_blocks(dit):
    """Once per model: pinned CPU master copies of every block tensor + where each one is bound."""
    if getattr(dit, "_lc_pinned", None) is not None:
        return dit._lc_pinned
    pinned = {}
    for i, blk in enumerate(dit.blocks):
        refs, ts = [], []
        for mod in blk.modules():
            for k, p in mod._parameters.items():
                if p is not None:
                    t = p.data.cpu()
                    t = t if t.is_pinned() else t.pin_memory()
                    p.data = t
                    refs.append((mod, k, True)); ts.append(t)
            for k, b in mod._buffers.items():
                if b is not None:
                    t = b.cpu()
                    t = t if t.is_pinned() else t.pin_memory()
                    mod._buffers[k] = t
                    refs.append((mod, k, False)); ts.append(t)
        pinned[i] = (refs, ts)
    dit._lc_pinned = pinned
    return pinned


class BlockStreamer:
    def __init__(self, dit, device, num_resident):
        self.dit = dit
        self.device = torch.device(device)
        n = len(dit.blocks)
        self.num_resident = max(0, min(n, int(num_resident)))
        self.resident = list(range(self.num_resident))
        self.streamed = list(range(self.num_resident, n))
        self.pinned = _pin_blocks(dit)
        self.block_bytes = sum(t.numel() * t.element_size() for t in self.pinned[0][1])
        self.hooks = []
        self.parked = True
        self.gpu_t = {}

        self._move_non_blocks(self.device)
        self.unpark()

        if self.streamed:
            tmpl = self.pinned[self.streamed[0]][1]
            for i in self.streamed:
                assert [(t.shape, t.dtype) for t in self.pinned[i][1]] == [(t.shape, t.dtype) for t in tmpl]
            self.slots = [[torch.empty(t.shape, dtype=t.dtype, device=self.device) for t in tmpl] for _ in range(2)]
            self.copy_stream = torch.cuda.Stream(device=self.device)
            self.slot_free = [None, None]
            self.pending = {}
            self.counter = 0
            self.next_of = {s: self.streamed[(k + 1) % len(self.streamed)] for k, s in enumerate(self.streamed)}
            for i in self.streamed:
                self.hooks.append(dit.blocks[i].register_forward_pre_hook(self._make_pre(i)))
                self.hooks.append(dit.blocks[i].register_forward_hook(self._make_post(i)))

    # ------------------------------------------------------------------ helpers
    def _move_non_blocks(self, device):
        dit = self.dit
        for name, child in dit.named_children():
            if name != "blocks":
                child.to(device)
        for k, p in dit._parameters.items():
            if p is not None:
                p.data = p.data.to(device)
        for k, b in dit._buffers.items():
            if b is not None:
                dit._buffers[k] = b.to(device)

    def _bind(self, i, tensors):
        for (mod, k, is_param), t in zip(self.pinned[i][0], tensors):
            if is_param:
                mod._parameters[k].data = t
            else:
                mod._buffers[k] = t

    def _schedule(self, i):
        if i in self.pending:
            return
        s = self.counter % 2
        self.counter += 1
        with torch.cuda.stream(self.copy_stream):
            if self.slot_free[s] is not None:
                self.copy_stream.wait_event(self.slot_free[s])
            for dst, src in zip(self.slots[s], self.pinned[i][1]):
                dst.copy_(src, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
        self.pending[i] = (s, ev)

    def _make_pre(self, i):
        def pre(module, args):
            self._schedule(i)
            s, ev = self.pending[i]
            torch.cuda.current_stream(self.device).wait_event(ev)
            self._bind(i, self.slots[s])
            self._schedule(self.next_of[i])  # prefetch the next streamed block into the other slot
        return pre

    def _make_post(self, i):
        def post(module, args, output):
            s, _ = self.pending.pop(i)
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.device))
            self.slot_free[s] = ev
            self._bind(i, self.pinned[i][1])
            return output
        return post

    # ------------------------------------------------------------------ park / unpark / close
    def park(self):
        """Drop the GPU copies of the resident blocks (masters stay in pinned RAM)."""
        if self.parked:
            return
        torch.cuda.synchronize(self.device)
        for i in self.resident:
            self._bind(i, self.pinned[i][1])
        self.gpu_t = {}
        if self.streamed:
            self.pending = {}
            for i in self.streamed:
                self._bind(i, self.pinned[i][1])
        self.parked = True
        torch.cuda.empty_cache()

    def unpark(self):
        if not self.parked:
            return
        for i in self.resident:
            g = [t.to(self.device, non_blocking=True) for t in self.pinned[i][1]]
            self.gpu_t[i] = g
            self._bind(i, g)
        torch.cuda.current_stream(self.device).synchronize()
        self.parked = False

    def close(self):
        self.park()
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.slots = None
        self._move_non_blocks("cpu")
        torch.cuda.empty_cache()

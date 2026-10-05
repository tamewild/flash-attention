# Copyright (c) 2026. Opt-in FA4 formed-operand GProj kernels.
# MMA/layout construction derives from FlashAttention (BSD-3-Clause), pin
# 843bf0b86bda1c92edc440ec58f9b5194609abac; see the accompanying guide.
"""Narrow Hopper auxiliary kernels for the opt-in public GProj path."""
import math
import operator
from typing import NamedTuple
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as hopper
from quack import layout_utils, sm90_utils
from flash_attn.cute import utils
from flash_attn.cute import copy_utils


class OperandProbes(NamedTuple):
    rows: cute.Tensor
    keys: cute.Tensor
    p: cute.Tensor
    g: cute.Tensor
    r: cute.Tensor
    t: cute.Tensor
    columns: cute.Tensor

class GProjTensors(NamedTuple):
    b: cute.Tensor
    mass: cute.Tensor
    rho: cute.Tensor
    probes: OperandProbes | None
    stats: cute.Tensor | None = None

@cute.jit
def collect_operands(gp, p, g, r, t, coords, mb, nb, head, length, wg):
    # Conversion preserved the scalar order of the original accumulators.
    rr = layout_utils.reshape_acc_to_mn(cute.make_tensor(r.iterator, p.layout))
    tt = layout_utils.reshape_acc_to_mn(cute.make_tensor(t.iterator, g.layout))
    pp, gg = layout_utils.reshape_acc_to_mn(p), layout_utils.reshape_acc_to_mn(g)
    lane = cute.arch.lane_idx()
    parts = cute.make_rmem_tensor((cute.size(rr, mode=[0]), 2), Float32)
    for i in cutlass.range_constexpr(cute.size(rr, mode=[0])):
        row = mb * 64 + coords[i, 0][0]
        rm, rt = Float32(0), Float32(0)
        for j in cutlass.range_constexpr(cute.size(rr, mode=[1])):
            rm += Float32(rr[i, j])
            rt += Float32(tt[i, j])
            if const_expr(gp.probes is not None):
                col = nb * 48 + coords[i, j][1]
                if row < length and col < length:
                    for z in cutlass.range(cute.size(gp.probes.rows), unroll=1):
                        if row == gp.probes.rows[z]:
                            gp.probes.p[z, head, col] = pp[i, j]
                            gp.probes.g[z, head, col] = gg[i, j]
                            gp.probes.r[z, head, col] = rr[i, j]
                            gp.probes.t[z, head, col] = tt[i, j]
                    for z in cutlass.range(cute.size(gp.probes.keys), unroll=1):
                        if col == gp.probes.keys[z]:
                            gp.probes.columns[z, head, row] = rr[i, j]
        rm = utils.warp_reduce(rm, operator.add, width=4)
        rt = utils.warp_reduce(rt, operator.add, width=4)
        parts[i, 0], parts[i, 1] = rm, rt
        if lane % 4 == 0 and wg == 1:
            # Defined padding and unique WG1 ownership for all 64 scratch rows.
            local_row = coords[i, 0][0]
            gp.stats[local_row] = rm if row < length else Float32(0)
            gp.stats[64 + local_row] = rt if row < length else Float32(0)
    return parts


@cute.jit
def finish_operands(gp, parts, coords, mb, head, length, wg,
                    deterministic: cutlass.Constexpr = False):
    # Called after the existing 256-thread PdS barrier, before its next barrier.
    if cute.arch.lane_idx() % 4 == 0 and wg == 0:
        for i in cutlass.range_constexpr(cute.size(parts, mode=[0])):
            local_row = coords[i, 0][0]
            row = mb * 64 + local_row
            if const_expr(deterministic):
                # WG0 publishes one combined packet. dQFullWG0 makes these
                # stores visible to the store warp. That warp consumes every
                # row before releasing either B-stage empty barrier, so neither
                # compute WG can reach the next iteration's scratch writes yet.
                gp.stats[local_row] = (parts[i, 0] + gp.stats[local_row]
                                       if row < length else Float32(0))
                gp.stats[64 + local_row] = (parts[i, 1] + gp.stats[64 + local_row]
                                            if row < length else Float32(0))
            elif row < length:
                utils.atomic_add_fp32(parts[i, 0] + gp.stats[local_row],
                                     gp.mass.iterator + gp.mass.layout((head, row)))
                utils.atomic_add_fp32(parts[i, 1] + gp.stats[64 + local_row],
                                     gp.rho.iterator + gp.rho.layout((head, row)))



def mma_q():
    return hopper.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16, warpgroup.OperandMajorMode.K,
        warpgroup.OperandMajorMode.MN, Float32, atom_layout_mnk=(1, 2, 1),
        tiler_mn=(64, 128))


def raw_copy():
    return cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
        cute.make_layout((128, 2)), cute.make_layout(4))


@cute.jit
def coalesced_store(mma, fragment, shared, output, block, tile_rows, swap):
    """Native R->S->R BF16 store pattern, with independent logical row tails."""
    tid = cute.arch.thread_idx()[0]
    atom = utils.get_smem_store_atom(90, cutlass.BFloat16, transpose=swap)
    r2s = cute.make_tiled_copy_C(atom, mma).get_slice(tid)
    dst = layout_utils.transpose_view(shared) if const_expr(swap) else shared
    cute.copy(r2s, r2s.retile(fragment), r2s.partition_D(dst))
    cute.arch.sync_threads()
    cp = copy_utils.tiled_copy_2d(cutlass.BFloat16, 256, 256)
    thr = cp.get_slice(tid)
    global_tile = cute.local_tile(output, (tile_rows, 256), (block, 0))
    gs, ss = thr.partition_S(global_tile), thr.partition_D(shared)
    rr = cute.make_fragment_like(ss, cutlass.BFloat16)
    cute.autovec_copy(ss, rr)
    coords = thr.partition_S(cute.make_identity_tensor((tile_rows, 256)))
    for m in cutlass.range_constexpr(cute.size(rr.shape[1])):
        if block * tile_rows + coords[0, m, 0][0] < output.shape[0]:
            cute.copy(cp, rr[None, m, None], gs[None, m, None])
    cute.arch.sync_threads()  # Every reader finishes before shared storage reuse.


class FusedQZ:
    @cute.jit
    def __call__(self, a, b, mass, rho, q, dq, z, rows, sample,
                 lam, stream: cuda.CUstream = None):
        shared = sm90_utils.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR,
                                            (64, 256), None, major_mode_size=128)
        self.kernel(a, b, mass, rho, q, dq, z, rows, sample, lam,
                    mma_q(), raw_copy(), shared).launch(
            grid=[cute.ceil_div(q.shape[1], 64), 16, 1], block=[256, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, a, b, mass, rho, q, dq, z, rows, sample, lam, mma, cp, shared):
        tid = cute.arch.thread_idx()[0]
        mb, head, _ = cute.arch.block_idx()
        coords = layout_utils.reshape_acc_to_mn(
            mma.get_slice(tid).partition_C(cute.make_identity_tensor((64, 256))))
        shape = mma.partition_shape_C((64, 256))
        aa, bb = cute.make_rmem_tensor(shape, Float32), cute.make_rmem_tensor(shape, Float32)
        for phase in cutlass.range_constexpr(2):
            source = a if const_expr(phase == 0) else b
            tile = cute.local_tile(source[0, head, None], (16384,), (mb,))
            tile = cute.make_tensor(tile.iterator, cute.make_layout((8192, 2)))
            src = cp.get_slice(tid).partition_D(tile)
            target = aa if const_expr(phase == 0) else bb
            cute.autovec_copy(src, cute.make_tensor(target.iterator, cute.make_layout(src.shape)))
        rdq, rz = cute.make_fragment_like(aa, cutlass.BFloat16), cute.make_fragment_like(aa, cutlass.BFloat16)
        amn, bmn = layout_utils.reshape_acc_to_mn(aa), layout_utils.reshape_acc_to_mn(bb)
        dmn, zmn = layout_utils.reshape_acc_to_mn(rdq), layout_utils.reshape_acc_to_mn(rz)
        for r in cutlass.range_constexpr(cute.size(amn, mode=[0])):
            row = mb * 64 + coords[r, 0][0]
            l = Float32(0)
            if row < q.shape[1]:
                if cute.arch.lane_idx() % 4 == 0:
                    m = mass[head, row]
                    if m != 0:
                        l = rho[head, row] / m
            # All warp lanes participate, including padded row groups. A full
            # shuffle member mask must not be used inside a divergent row tail.
            l = cute.arch.shuffle_sync(l, cute.arch.lane_idx() // 4 * 4)
            if const_expr(lam is not None):
                if row < q.shape[1]:
                    if cute.arch.lane_idx() % 4 == 0 and tid < 128:
                        lam[head, row] = l
            for c in cutlass.range_constexpr(cute.size(amn, mode=[1])):
                d = coords[r, c][1]
                corrected = (amn[r, c] - l * bmn[r, c]) * Float32(1 / 16)
                dmn[r, c] = corrected.to(cutlass.BFloat16)
                zv = Float32(0)
                if row < q.shape[1]:
                    zv = l * Float32(q[0, row, head, d])
                zmn[r, c] = zv.to(cutlass.BFloat16)
                if const_expr(sample is not None):
                    if row < q.shape[1]:
                        for j in cutlass.range(cute.size(rows), unroll=1):
                            if row == rows[j]:
                                sample[0, j, head, d] = amn[r, c]
                                sample[1, j, head, d] = bmn[r, c]
                                sample[2, j, head, d] = corrected
        alloc = cutlass.utils.SmemAllocator()
        sd = alloc.allocate_tensor(cutlass.BFloat16, shared.outer, byte_alignment=1024, swizzle=shared.inner)
        coalesced_store(mma, rdq, sd, dq[0, None, head, None], mb, 64, False)
        coalesced_store(mma, rz, sd, z[0, None, head, None], mb, 64, False)


@cute.jit
def copy_rows(source, shared, first, head, cp):
    """Aligned 16-byte copies; false rows receive explicit shared zeros."""
    tid = cute.arch.thread_idx()[0]
    tile_rows = cute.size(shared, mode=[0])
    tile = cute.local_tile(source[0, None, head, None], (tile_rows, 256), (first, 0))
    thr = cp.get_slice(tid)
    src, dst = thr.partition_S(tile), thr.partition_D(shared)
    coords = thr.partition_S(cute.make_identity_tensor((tile_rows, 256)))
    zero_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
    for m in cutlass.range_constexpr(cute.size(src.shape[1])):
        if first * tile_rows + coords[0, m, 0][0] < source.shape[1]:
            cute.copy(cp, src[None, m, None], dst[None, m, None])
        else:
            zero = cute.make_fragment_like(dst[None, m, None], cutlass.BFloat16)
            zero.fill(0)
            cute.copy(zero_atom, zero, dst[None, m, None])


class KeyCorrection:
    """Matching 64x48 score MMA; selectable one/two-stage Q/Z copies.

    K/P remain single buffered. Matrix waits, copy waits, and CTA visibility
    are separate. The correction is subtracted/scaled/cast directly to BF16.
    """
    def __init__(self, stages=2):
        if stages not in (1, 2):
            raise ValueError("Key stages must be one or two")
        self.stages = stages

    @cute.jit
    def __call__(self, q, k, z, lse, dk_acc, dk, keys, columns, sample, stream: cuda.CUstream = None):
        sq = sm90_utils.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR,
                                        (64, 256), self.stages, major_mode_size=128)
        sk = sm90_utils.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR,
                                        (48, 256), None, major_mode_size=128)
        sp = sm90_utils.make_smem_layout(cutlass.BFloat16, LayoutEnum.ROW_MAJOR,
                                        (64, 48), None, major_mode_size=24)
        score = hopper.make_trivial_tiled_mma(cutlass.BFloat16, cutlass.BFloat16,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K, Float32,
            atom_layout_mnk=(1, 2, 1), tiler_mn=(64, 24))
        product = hopper.make_trivial_tiled_mma(cutlass.BFloat16, cutlass.BFloat16,
            warpgroup.OperandMajorMode.MN, warpgroup.OperandMajorMode.MN, Float32,
            atom_layout_mnk=(2, 1, 1), tiler_mn=(64, 48))
        cp = copy_utils.tiled_copy_2d(cutlass.BFloat16, 256, 256, is_async=True)
        self.kernel(q, k, z, lse, dk_acc, dk, keys, columns, sample, sq, sk, sp,
                    score, product, raw_copy(), cp).launch(
            grid=[cute.ceil_div(k.shape[1], 48), 4, 1], block=[256, 1, 1], stream=stream)

    @cute.jit
    def math_tile(self, q, lse, sQ, sK, sZ, sP, accC, score, product,
                  mb, nb, head, keys, columns):
        tid = cute.arch.thread_idx()[0]
        wg = (tid // 128) * 128
        _, fragQ, fragK = sm90_utils.partition_fragment_ABC(score.get_slice(wg), (64, 48, 256), sQ, sK)
        _, fragP, fragZ = sm90_utils.partition_fragment_ABC(product.get_slice(wg), (48, 256, 64),
            layout_utils.transpose_view(sP), layout_utils.transpose_view(sZ), swap_AB=True)
        accS = sm90_utils.gemm_zero_init(score, (64, 48), fragQ, fragK, wg_wait=0)
        coords = score.get_slice(tid).partition_C(cute.make_identity_tensor((64, 48)))
        for j in cutlass.range_constexpr(cute.size(accS)):
            row, key = mb * 64 + coords[j][0], nb * 48 + coords[j][1]
            p = Float32(0)
            if row < q.shape[1] and key < q.shape[1] and key <= row:
                p = cute.math.exp2(accS[j] * Float32(math.log2(math.e) / 16) - lse[0, head, row], fastmath=True)
            r = p.to(cutlass.BFloat16)
            sP[coords[j][0], coords[j][1]] = r
            if const_expr(columns is not None):
                if row < q.shape[1] and key < q.shape[1]:
                    for jkey in cutlass.range(cute.size(keys), unroll=1):
                        if key == keys[jkey]:
                            columns[jkey, head, row] = r
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        sm90_utils.gemm_w_idx(product, accC, fragP, fragZ, zero_init=False, swap_AB=True, wg_wait=0)
        cute.arch.sync_threads()  # Both WGs have drained all P/Z readers.

    @cute.kernel
    def kernel(self, q, k, z, lse, dk_acc, dk, keys, columns, sample, sq, sk, sp, score, product, raw_cp, cp):
        tid = cute.arch.thread_idx()[0]
        nb, kvhead, _ = cute.arch.block_idx()
        alloc = cutlass.utils.SmemAllocator()
        sQ = alloc.allocate_tensor(cutlass.BFloat16, sq.outer, byte_alignment=1024, swizzle=sq.inner)
        sK = alloc.allocate_tensor(cutlass.BFloat16, sk.outer, byte_alignment=1024, swizzle=sk.inner)
        sZ = alloc.allocate_tensor(cutlass.BFloat16, sq.outer, byte_alignment=1024, swizzle=sq.inner)
        sP = alloc.allocate_tensor(cutlass.BFloat16, sp.outer, byte_alignment=1024, swizzle=sp.inner)
        accC = cute.make_rmem_tensor(product.partition_shape_C((256, 48)), Float32)
        accC.fill(0)
        copy_rows(k, sK, nb, kvhead, cp)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        first, stop = (nb * 48) // 64, cute.ceil_div(q.shape[1], 64)
        for gh in cutlass.range_constexpr(4):
            head = kvhead * 4 + gh
            slot = Int32(0)
            if const_expr(self.stages == 2):
                copy_rows(q, sQ[None, None, 0], first, head, cp)
                copy_rows(z, sZ[None, None, 0], first, head, cp)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_threads()
            for mb in cutlass.range(first, stop, unroll=1):
                if const_expr(self.stages == 2):
                    if mb + 1 < stop:
                        copy_rows(q, sQ[None, None, slot ^ 1], mb + 1, head, cp)
                        copy_rows(z, sZ[None, None, slot ^ 1], mb + 1, head, cp)
                        cute.arch.cp_async_commit_group()
                else:
                    copy_rows(q, sQ[None, None, 0], mb, head, cp)
                    copy_rows(z, sZ[None, None, 0], mb, head, cp)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_threads()
                self.math_tile(q, lse, sQ[None, None, slot], sK, sZ[None, None, slot], sP,
                               accC, score, product, mb, nb, head, keys, columns)
                if const_expr(self.stages == 2):
                    if mb + 1 < stop:
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        cute.arch.sync_threads()
                        slot = slot ^ 1
        tile = cute.local_tile(dk_acc[0, kvhead, None], (12288,), (nb,))
        tile = cute.make_tensor(tile.iterator, cute.make_layout((6144, 2)))
        raw = raw_cp.get_slice(tid).partition_D(tile)
        original = cute.make_rmem_tensor(accC.shape, Float32)
        cute.autovec_copy(raw, cute.make_tensor(original.iterator, cute.make_layout(raw.shape)))
        coords = product.get_slice(tid).partition_C(cute.make_identity_tensor((256, 48)))
        rd = cute.make_fragment_like(accC, cutlass.BFloat16)
        for i in cutlass.range_constexpr(cute.size(accC)):
            corrected = original[i] - accC[i]
            rd[i] = (corrected * Float32(1 / 16)).to(cutlass.BFloat16)
            if const_expr(sample is not None):
                key, d = nb * 48 + coords[i][1], coords[i][0]
                for j in cutlass.range(cute.size(keys), unroll=1):
                    if key == keys[j] and key < k.shape[1]:
                        sample[0, j, kvhead, d] = original[i]
                        sample[1, j, kvhead, d] = accC[i]
                        sample[2, j, kvhead, d] = corrected
        # All WGMMA readers drained above. K is dead and becomes the BF16 output stage.
        coalesced_store(product, rd, sK, dk[0, None, kvhead, None], nb, 48, True)

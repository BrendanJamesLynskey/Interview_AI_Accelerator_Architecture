# Worked Problem: Collective Communication

## Problem Statement

You have 32 GPUs arranged in 4 nodes of 8 GPUs each. Within each node, GPUs are connected by NVLink at 900 GB/s bidirectional all-to-all. Between nodes, each GPU has one InfiniBand port at 50 GB/s.

You need to perform an all-reduce of a 4 GB gradient tensor (BF16, 2B parameter model).

Compare:
1. Flat ring all-reduce across all 32 GPUs
2. Hierarchical all-reduce (intra-node ring, then inter-node ring, then intra-node broadcast)
3. Determine which approach is faster and by how much

---

## Worked Solution

### Step 1: Flat ring all-reduce

In a flat ring across 32 GPUs, the ring must traverse both NVLink (within nodes) and InfiniBand (between nodes). The bottleneck link determines the effective bandwidth.

With 4 nodes in a ring, each message traverses 3 intra-node NVLink hops and 4 inter-node InfiniBand hops (8->node boundary->8->boundary->...). The ring bandwidth is limited by the slowest link: InfiniBand at 50 GB/s.

```
Ring all-reduce time = 2 * (31/32) * 4 GB / 50 GB/s = 155 ms
```

Actually, in a ring all-reduce, each GPU sends (N-1)/N of the data in each of 2 phases (reduce-scatter + all-gather). The total data sent per GPU per phase is (N-1)/N * data_size. The time is determined by the bandwidth of the link each GPU sends on.

In the flat ring, some links are NVLink (900 GB/s) and some are InfiniBand (50 GB/s). Each GPU sends data on one link. If the ring alternates between intra-node and inter-node links, each GPU either sends on NVLink or InfiniBand. The 4 GPUs at node boundaries send on InfiniBand (50 GB/s), and the 28 interior GPUs send on NVLink (900 GB/s).

The ring's throughput is limited by the slowest link. Time for reduce-scatter phase:
```
Data per phase = (31/32) * 4 GB = 3.875 GB
Time = 3.875 GB / 50 GB/s = 77.5 ms (bottlenecked by InfiniBand links)
```

Total (both phases): 2 * 77.5 = 155 ms

### Step 2: Hierarchical all-reduce

**Phase 1: Intra-node reduce-scatter** (8 GPUs per node, NVLink)
```
Data per GPU = (7/8) * 4 GB = 3.5 GB
Time = 3.5 / 900 = 3.89 ms
```
After this, each GPU holds 1/8 of the partially reduced result (0.5 GB per GPU).

**Phase 2: Inter-node all-reduce** of each GPU's 0.5 GB portion (4 nodes, InfiniBand)
Each GPU communicates with its corresponding GPU in other nodes (GPU i in node 0 talks to GPU i in nodes 1-3).
```
Ring all-reduce across 4 nodes:
Data per phase = (3/4) * 0.5 GB = 0.375 GB
Time = 2 * 0.375 / 50 = 15 ms
```

**Phase 3: Intra-node all-gather** (broadcast the result within each node)
```
Data per GPU = (7/8) * 4 GB = 3.5 GB
Time = 3.5 / 900 = 3.89 ms
```

**Total hierarchical time**: 3.89 + 15.0 + 3.89 = 22.78 ms

### Step 3: Comparison

| Method | Time | Speedup |
|---|---|---|
| Flat ring | 155 ms | 1.0x |
| Hierarchical | 22.78 ms | 6.8x |

### Step 4: Analysis

The hierarchical approach is 6.8x faster because:

1. **Intra-node phases use NVLink**: The 900 GB/s NVLink bandwidth is 18x faster than InfiniBand. The intra-node phases (reduce-scatter and all-gather) complete in under 4 ms each.

2. **Inter-node phase has reduced data**: After the intra-node reduce-scatter, each GPU holds only 1/8 of the data. The inter-node communication volume is 8x less than the flat ring.

3. **Fewer inter-node hops**: Only 4 nodes participate in the inter-node ring (vs 32 GPUs in the flat ring), reducing the number of InfiniBand traversals.

The hierarchical approach is optimal when there is a bandwidth hierarchy (fast intra-node, slow inter-node). NCCL automatically selects hierarchical algorithms when it detects such topology. This is why modern GPU clusters are designed with NVLink intra-node and InfiniBand inter-node: the hierarchical communication strategy leverages the high intra-node bandwidth to reduce inter-node traffic.

### Step 5: Practical considerations

- **Latency overhead**: Each phase has startup latency (connection setup, synchronization). The hierarchical approach has 3 phases vs 1, adding approximately 30-50 us of overhead. Negligible compared to the ms-scale data transfer times.
- **Computation overlap**: Both approaches can overlap communication with backward pass computation. The hierarchical approach's shorter total time means less computation is needed to overlap.
- **SHARP acceleration**: InfiniBand SHARP can perform in-network reduction during Phase 2, potentially halving the inter-node communication time.

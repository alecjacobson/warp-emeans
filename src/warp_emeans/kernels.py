"""Implement assignment, EMA, shuffling and donor revival in Warp."""

from functools import cache

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def validate_kernel(x: wp.array2d[float], invalid: wp.array[int]):
    row, column = wp.tid()
    if not wp.isfinite(x[row, column]):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def shuffle_kernel(keys: wp.array[wp.int32], order: wp.array[int], seed: int):
    row = wp.tid()
    state = wp.rand_init(seed, row)
    keys[row] = wp.randi(state)
    order[row] = row


@wp.kernel
def initialize_kernel(
    x: wp.array2d[float], order: wp.array[int], centers: wp.array2d[float], usage: wp.array[float]
):
    cluster, column = wp.tid()
    centers[cluster, column] = x[order[cluster], column]
    if column == 0:
        usage[cluster] = 0.0000152587890625


@cache
def assignment_kernel(d):
    """Specialize direct squared-distance assignment on vector dimension."""
    vector = wp.types.vector(length=d, dtype=wp.float32)

    @wp.kernel(enable_backward=False)
    def assign_kernel(
        x: wp.array2d[float],
        centers: wp.array2d[float],
        order: wp.array[int],
        offset: int,
        k: int,
        sums: wp.array2d[float],
        counts: wp.array[int],
        labels: wp.array[int],
        distances: wp.array[float],
        accumulate: int,
        indirect: int,
    ):
        local = wp.tid()
        row = local
        if indirect != 0:
            row = order[offset + local]
        point = vector()
        for column in range(wp.static(d)):
            point[column] = x[row, column]
        best = float(3.4028234663852886e38)
        label = int(0)
        for cluster in range(k):
            distance = float(0.0)
            for column in range(wp.static(d)):
                delta = point[column] - centers[cluster, column]
                distance += delta * delta
            if distance < best:
                best = distance
                label = cluster
        labels[row] = label
        distances[row] = best
        if accumulate != 0:
            wp.atomic_add(counts, label, 1)
            for column in range(wp.static(d)):
                wp.atomic_add(sums, label, column, point[column])

    return assign_kernel


@wp.kernel
def update_kernel(
    centers: wp.array2d[float],
    usage: wp.array[float],
    sums: wp.array2d[float],
    counts: wp.array[int],
    closing_sums: wp.array2d[float],
    closing_counts: wp.array[int],
    schedule: float,
    peak: wp.array[float],
    closing: int,
):
    cluster = wp.tid()
    alpha = schedule * peak[0]
    count = counts[cluster]
    old_mass = usage[cluster]
    mass = (1.0 - alpha) * old_mass + alpha * float(count)
    # Preserve unused mass when alpha=1, matching the pinned reference.
    if count == 0 and alpha == 1.0:
        mass = old_mass
    usage[cluster] = mass
    if closing != 0:
        closing_counts[cluster] += count
    for column in range(centers.shape[1]):
        value = sums[cluster, column]
        if count > 0:
            mean = value / float(count)
            weight = alpha * float(count) / mass
            position = centers[cluster, column] + weight * (mean - centers[cluster, column])
            if weight == 1.0:
                position = mean
            centers[cluster, column] = position
        if closing != 0:
            closing_sums[cluster, column] += value
        sums[cluster, column] = 0.0
    counts[cluster] = 0


@wp.kernel
def candidates_kernel(
    usage: wp.array[float],
    total: wp.array[float],
    threshold: float,
    keys: wp.array[float],
    donors: wp.array[int],
    dead: wp.array[int],
    seed: int,
):
    cluster = wp.tid()
    mass = usage[cluster]
    cutoff = threshold * total[0] / float(usage.shape[0])
    dead[cluster] = int(mass < cutoff)
    donors[cluster] = cluster
    keys[cluster] = 3.4028234663852886e38
    if mass >= cutoff and mass > 0.0:
        state = wp.rand_init(seed, cluster)
        uniform = wp.max(wp.randf(state), 1e-20)
        keys[cluster] = -(wp.log(mass) - wp.log(-wp.log(uniform)))


@wp.kernel
def split_kernel(
    centers: wp.array2d[float],
    usage: wp.array[float],
    donors: wp.array[int],
    keys: wp.array[float],
    dead: wp.array[int],
    prefix: wp.array[int],
    splits: wp.array[int],
):
    target = wp.tid()
    if dead[target] != 0:
        rank = prefix[target] - 1
        if keys[rank] < 3.4028234663852886e38:
            source = donors[rank]
            half = usage[source] * 0.5
            usage[source] = half
            usage[target] = half
            different = bool(False)
            for column in range(centers.shape[1]):
                offset = 0.0001
                if column % 2 == 1:
                    offset = -offset
                value = centers[source, column]
                donor_value = value * (1.0 - offset)
                child_value = value * (1.0 + offset)
                centers[source, column] = donor_value
                centers[target, column] = child_value
                different = different or donor_value != child_value
            if not different:
                # The smallest positive FP32 value separates a zero donor, as upstream does.
                centers[target, 0] = wp.float32(1.401298464324817e-45)
            wp.atomic_add(splits, 0, 1)


@wp.kernel
def finalize_kernel(centers: wp.array2d[float], sums: wp.array2d[float], counts: wp.array[int]):
    cluster, column = wp.tid()
    if counts[cluster] > 0:
        centers[cluster, column] = sums[cluster, column] / float(counts[cluster])


@cache
def cooperative_assignment_kernel(d):
    """Distribute each point's centroid search across one GPU warp."""
    vector = wp.types.vector(length=d, dtype=wp.float32)

    @wp.kernel(enable_backward=False)
    def cooperative_assign_kernel(
        x: wp.array2d[float],
        centers: wp.array2d[float],
        order: wp.array[int],
        offset: int,
        k: int,
        sums: wp.array2d[float],
        counts: wp.array[int],
        labels: wp.array[int],
        distances: wp.array[float],
        accumulate: int,
        indirect: int,
    ):
        local, lane = wp.tid()
        row = local
        if indirect != 0:
            row = order[offset + local]
        point = vector()
        for column in range(wp.static(d)):
            point[column] = x[row, column]
        best = float(3.4028234663852886e38)
        label = int(2147483647)
        for cluster in range(lane, k, 32):
            distance = float(0.0)
            for column in range(wp.static(d)):
                delta = point[column] - centers[cluster, column]
                distance += delta * delta
            if distance < best:
                best = distance
                label = cluster
        minimum = wp.tile_min(wp.tile(best))[0]
        candidate = int(2147483647)
        if best == minimum:
            candidate = label
        winner = wp.tile_min(wp.tile(candidate))[0]
        if lane == 0:
            labels[row] = winner
            distances[row] = minimum
            if accumulate != 0:
                wp.atomic_add(counts, winner, 1)
                for column in range(wp.static(d)):
                    wp.atomic_add(sums, winner, column, point[column])

    return cooperative_assign_kernel

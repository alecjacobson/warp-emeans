"""Implement full-data Lloyd assignment and centroid reduction in Warp."""

from functools import cache

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def validate_magnitude_kernel(x: wp.array2d[float], limit: float, invalid: wp.array[int]):
    row, column = wp.tid()
    if wp.abs(x[row, column]) > limit:
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def initialize_kernel(x: wp.array2d[float], order: wp.array[int], centers: wp.array2d[float]):
    cluster, column = wp.tid()
    centers[cluster, column] = x[order[cluster], column]


@cache
def scalar_assignment_kernel(d, unroll=False):
    """Specialize one-thread-per-point direct squared-distance assignment."""
    vector = wp.types.vector(length=d, dtype=wp.float32)

    @wp.kernel(
        enable_backward=False, module="unique", module_options={"max_unroll": max(16, d) if unroll else 16}
    )
    def scalar_assign_kernel(
        x: wp.array2d[float],
        centers: wp.array2d[float],
        sums: wp.array3d[float],
        counts: wp.array2d[int],
        labels: wp.array[int],
        distances: wp.array[float],
        accumulate: int,
    ):
        row = wp.tid()
        point = vector()
        for column in range(wp.static(d)):
            point[column] = x[row, column]
        best = float(wp.inf)
        winner = int(0)
        for cluster in range(centers.shape[0]):
            distance = float(0.0)
            for column in range(wp.static(d)):
                delta = point[column] - centers[cluster, column]
                distance += delta * delta
            if distance < best:
                best = distance
                winner = cluster
        labels[row] = winner
        distances[row] = best
        if accumulate != 0:
            shard = (row // 256) % counts.shape[0]
            wp.atomic_add(counts, shard, winner, 1)
            for column in range(wp.static(d)):
                wp.atomic_add(sums, shard, winner, column, point[column])

    return scalar_assign_kernel


@wp.kernel
def accumulate_kernel(
    x: wp.array2d[float], labels: wp.array[int], sums: wp.array3d[float], counts: wp.array2d[int]
):
    row, column = wp.tid()
    cluster = labels[row]
    shard = (row // 256) % counts.shape[0]
    wp.atomic_add(sums, shard, cluster, column, x[row, column])
    if column == 0:
        wp.atomic_add(counts, shard, cluster, 1)


@wp.kernel
def update_kernel(centers: wp.array2d[float], sums: wp.array3d[float], counts: wp.array2d[int]):
    cluster, column = wp.tid()
    total = float(0.0)
    count = int(0)
    for shard in range(counts.shape[0]):
        total += sums[shard, cluster, column]
        count += counts[shard, cluster]
    if count > 0:
        centers[cluster, column] = total / float(count)


@wp.kernel
def movement_kernel(previous: wp.array2d[float], centers: wp.array2d[float], maximum: wp.array[wp.float64]):
    cluster = wp.tid()
    squared = wp.float64(0.0)
    for column in range(centers.shape[1]):
        delta = wp.float64(centers[cluster, column]) - wp.float64(previous[cluster, column])
        squared += delta * delta
    # Round outward relative to FP64 arithmetic before relaxing the bounds.
    movement = wp.sqrt(squared) * wp.float64(1.0 + 1e-12) + wp.float64(1e-30)
    wp.atomic_max(maximum, 0, movement)


@cache
def bounded_assignment_kernel(d):
    """Skip a full search only when conservative bounds separate the winner."""
    vector = wp.types.vector(length=d, dtype=wp.float32)
    relative_guard = 16.0 * d * 2.0**-23
    absolute_guard = d * 2.0**-60

    @wp.kernel(enable_backward=False, module="unique", module_options={"max_unroll": max(16, d)})
    def bounded_assign_kernel(
        x: wp.array2d[float],
        centers: wp.array2d[float],
        sums: wp.array3d[float],
        counts: wp.array2d[int],
        labels: wp.array[int],
        distances: wp.array[float],
        lower_bounds: wp.array[wp.float64],
        maximum_movement: wp.array[wp.float64],
        searches: wp.array[int],
    ):
        row = wp.tid()
        point = vector()
        previous_label = labels[row]
        previous_squared = float(0.0)
        for column in range(wp.static(d)):
            point[column] = x[row, column]
            delta = point[column] - centers[previous_label, column]
            previous_squared += delta * delta
        guard = wp.float64(wp.static(relative_guard))
        tiny = wp.float64(wp.static(absolute_guard))
        lower = wp.max(
            wp.float64(0.0),
            (lower_bounds[row] - maximum_movement[0]) * wp.float64(1.0 - 1e-12) - wp.float64(1e-30),
        )
        upper = wp.sqrt(wp.float64(previous_squared)) * (wp.float64(1.0) + guard) + tiny
        winner = previous_label
        best = previous_squared
        # The extra guard protects the FP32 distance comparison, including ties.
        if upper * (wp.float64(1.0) + guard) + tiny >= lower * (wp.float64(1.0) - guard) - tiny:
            best = float(wp.inf)
            second = float(wp.inf)
            winner = int(0)
            for cluster in range(centers.shape[0]):
                distance = float(0.0)
                for column in range(wp.static(d)):
                    delta = point[column] - centers[cluster, column]
                    distance += delta * delta
                if distance < best:
                    second = best
                    best = distance
                    winner = cluster
                elif distance < second:
                    second = distance
            lower = wp.max(wp.float64(0.0), wp.sqrt(wp.float64(second)) * (wp.float64(1.0) - guard) - tiny)
            searches[row] += 1
        lower_bounds[row] = lower
        labels[row] = winner
        distances[row] = best
        shard = (row // 256) % counts.shape[0]
        wp.atomic_add(counts, shard, winner, 1)
        for column in range(wp.static(d)):
            wp.atomic_add(sums, shard, winner, column, point[column])

    return bounded_assign_kernel


@cache
def screen_kernel(d):
    """Mark points whose conservative second-center bound requires a search."""

    @wp.kernel(enable_backward=False, module="unique", module_options={"max_unroll": max(16, d)})
    def screen_points_kernel(
        x: wp.array2d[float],
        centers: wp.array2d[float],
        labels: wp.array[int],
        distances: wp.array[float],
        lower_bounds: wp.array[wp.float64],
        maximum: wp.array[wp.float64],
        flags: wp.array[int],
    ):
        row = wp.tid()
        distance = float(0.0)
        for column in range(wp.static(d)):
            delta = x[row, column] - centers[labels[row], column]
            distance += delta * delta
        distances[row] = distance
        guard = wp.float64(wp.static(16.0 * d * 2.0**-23))
        tiny = wp.float64(wp.static(d * 2.0**-60))
        lower = wp.max(
            wp.float64(0.0), (lower_bounds[row] - maximum[0]) * wp.float64(1.0 - 1e-12) - wp.float64(1e-30)
        )
        lower_bounds[row] = lower
        upper = wp.sqrt(wp.float64(distance)) * (wp.float64(1.0) + guard) + tiny
        flags[row] = int(upper * (wp.float64(1.0) + guard) + tiny >= lower * (wp.float64(1.0) - guard) - tiny)

    return screen_points_kernel


@wp.kernel
def compact_kernel(flags: wp.array[int], prefix: wp.array[int], active: wp.array[int]):
    row = wp.tid()
    if flags[row] != 0:
        active[prefix[row] - 1] = row


@cache
def compact_assignment_kernel(d):
    """Search compacted active points and retain conservative second distances."""
    vector = wp.types.vector(length=d, dtype=wp.float32)

    @wp.kernel(enable_backward=False, module="unique", module_options={"max_unroll": max(16, d)})
    def compact_assign_kernel(
        x: wp.array2d[float],
        centers: wp.array2d[float],
        labels: wp.array[int],
        distances: wp.array[float],
        lower_bounds: wp.array[wp.float64],
        searches: wp.array[int],
        active: wp.array[int],
        prefix: wp.array[int],
    ):
        local = wp.tid()
        if local < prefix[prefix.shape[0] - 1]:
            row = active[local]
            point = vector()
            for column in range(wp.static(d)):
                point[column] = x[row, column]
            best = float(wp.inf)
            second = float(wp.inf)
            winner = int(0)
            for cluster in range(centers.shape[0]):
                distance = float(0.0)
                for column in range(wp.static(d)):
                    delta = point[column] - centers[cluster, column]
                    distance += delta * delta
                if distance < best:
                    second = best
                    best = distance
                    winner = cluster
                elif distance < second:
                    second = distance
            guard = wp.float64(wp.static(16.0 * d * 2.0**-23))
            tiny = wp.float64(wp.static(d * 2.0**-60))
            lower_bounds[row] = wp.max(
                wp.float64(0.0), wp.sqrt(wp.float64(second)) * (wp.float64(1.0) - guard) - tiny
            )
            searches[row] += 1
            labels[row] = winner
            distances[row] = best

    return compact_assign_kernel

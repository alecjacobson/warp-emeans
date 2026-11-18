"""Implement full-data Lloyd assignment and centroid reduction in Warp."""

from functools import cache

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def initialize_kernel(x: wp.array2d[float], order: wp.array[int], centers: wp.array2d[float]):
    cluster, column = wp.tid()
    centers[cluster, column] = x[order[cluster], column]


@cache
def scalar_assignment_kernel(d):
    """Specialize one-thread-per-point direct squared-distance assignment."""
    vector = wp.types.vector(length=d, dtype=wp.float32)

    @wp.kernel(enable_backward=False)
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


@cache
def cooperative_assignment_kernel(d):
    """Distribute a point's centroid search across a 32-thread warp."""
    vector = wp.types.vector(length=d, dtype=wp.float32)

    @wp.kernel(enable_backward=False)
    def cooperative_assign_kernel(
        x: wp.array2d[float],
        centers: wp.array2d[float],
        sums: wp.array3d[float],
        counts: wp.array2d[int],
        labels: wp.array[int],
        distances: wp.array[float],
        accumulate: int,
    ):
        row, lane = wp.tid()
        point = vector()
        for column in range(wp.static(d)):
            point[column] = x[row, column]
        best = float(wp.inf)
        winner = int(2147483647)
        for cluster in range(lane, centers.shape[0], 32):
            distance = float(0.0)
            for column in range(wp.static(d)):
                delta = point[column] - centers[cluster, column]
                distance += delta * delta
            if distance < best or (distance == best and cluster < winner):
                best = distance
                winner = cluster
        minimum = wp.tile_min(wp.tile(best))[0]
        candidate = int(2147483647)
        if best == minimum:
            candidate = winner
        label = wp.tile_min(wp.tile(candidate))[0]
        if lane == 0:
            labels[row] = label
            distances[row] = minimum
            if accumulate != 0:
                shard = (row // 256) % counts.shape[0]
                wp.atomic_add(counts, shard, label, 1)
                for column in range(wp.static(d)):
                    wp.atomic_add(sums, shard, label, column, point[column])

    return cooperative_assign_kernel


@wp.kernel
def accumulate_kernel(x: wp.array2d[float], labels: wp.array[int], sums: wp.array3d[float], counts: wp.array2d[int]):
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


@wp.func
def choose_label(distance: float, minimum: float, cluster: int):
    result = int(2147483647)
    if distance == minimum:
        result = cluster
    return result


@wp.func
def merge_label(distance: float, best: float, cluster: int, label: int):
    result = label
    if distance < best:
        result = cluster
    return result


@cache
def tiled_assignment_kernel(d, point_tile=32, center_tile=32):
    """Reuse coordinate tiles across a rectangular set of point-center pairs."""

    @wp.func
    def mask_distance(distance: float, cluster: int, k: int):
        value = distance
        if cluster >= k:
            value = wp.inf
        return value

    @wp.kernel(enable_backward=False)
    def tiled_assign_kernel(
        x: wp.array2d[float],
        centers: wp.array2d[float],
        labels: wp.array[int],
        distances: wp.array[float],
    ):
        block = wp.tid()
        best = wp.tile_full(shape=(point_tile,), value=wp.inf, dtype=float)
        winners = wp.tile_zeros(shape=(point_tile,), dtype=int)
        for start in range(0, centers.shape[0], center_tile):
            squared = wp.tile_zeros(shape=(point_tile, center_tile), dtype=float)
            for column in range(wp.static(d)):
                points = wp.tile_load(x, shape=(point_tile, 1), offset=(block * point_tile, column))
                coordinates = wp.tile_load(centers, shape=(center_tile, 1), offset=(start, column))
                delta = wp.tile_broadcast(points, shape=(point_tile, center_tile)) - wp.tile_broadcast(
                    wp.tile_transpose(coordinates), shape=(point_tile, center_tile)
                )
                squared = squared + wp.tile_map(wp.mul, delta, delta)
            ids = wp.tile_map(wp.add, wp.tile_arange(center_tile, dtype=int), start)
            ids_matrix = wp.tile_broadcast(wp.tile_reshape(ids, shape=(1, center_tile)), shape=(point_tile, center_tile))
            valid = wp.tile_map(mask_distance, squared, ids_matrix, centers.shape[0])
            minima = wp.tile_reduce(wp.min, valid, axis=1)
            minimum_matrix = wp.tile_broadcast(wp.tile_reshape(minima, shape=(point_tile, 1)), shape=(point_tile, center_tile))
            candidates = wp.tile_map(choose_label, valid, minimum_matrix, ids_matrix)
            nearest = wp.tile_reduce(wp.min, candidates, axis=1)
            winners = wp.tile_map(merge_label, minima, best, nearest, winners)
            best = wp.tile_map(wp.min, best, minima)
        wp.tile_store(labels, winners, offset=block * point_tile)
        wp.tile_store(distances, best, offset=block * point_tile)

    return tiled_assign_kernel

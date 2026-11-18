"""Retain measured experimental assignment layouts; not used by Kmeans."""

from functools import cache

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.struct
class DistanceInputs:
    points: wp.array2d[float]
    centers: wp.array2d[float]


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
            ids_matrix = wp.tile_broadcast(
                wp.tile_reshape(ids, shape=(1, center_tile)), shape=(point_tile, center_tile)
            )
            valid = wp.tile_map(mask_distance, squared, ids_matrix, centers.shape[0])
            minima = wp.tile_reduce(wp.min, valid, axis=1)
            minimum_matrix = wp.tile_broadcast(
                wp.tile_reshape(minima, shape=(point_tile, 1)), shape=(point_tile, center_tile)
            )
            candidates = wp.tile_map(choose_label, valid, minimum_matrix, ids_matrix)
            nearest = wp.tile_reduce(wp.min, candidates, axis=1)
            winners = wp.tile_map(merge_label, minima, best, nearest, winners)
            best = wp.tile_map(wp.min, best, minima)
        wp.tile_store(labels, winners, offset=block * point_tile)
        wp.tile_store(distances, best, offset=block * point_tile)

    return tiled_assign_kernel


@cache
def matmul_assignment_kernel(d, point_tile=16, center_tile=32):
    """Screen pairs with Tensor Cores and refine possible winners in direct FP32."""
    width = 2 ** (d - 1).bit_length()
    error_factor = 8.0 * 2.0**-10 + 16.0 * d * 2.0**-23

    @wp.func
    def approximate_distance(dot: float, point_norm: float, center_norm: float):
        return point_norm + center_norm - 2.0 * dot

    @wp.func
    def upper_distance(approximate: float, point_norm: float, center_norm: float):
        return approximate + wp.static(error_factor) * (point_norm + center_norm) + wp.static(d * 2.0**-120)

    @wp.func
    def refine_distance(
        approximate: float,
        point_norm: float,
        center_norm: float,
        limit: float,
        row: int,
        cluster: int,
        inputs: DistanceInputs,
    ):
        value = float(wp.inf)
        error = wp.static(error_factor) * (point_norm + center_norm) + wp.static(d * 2.0**-120)
        if (
            row < inputs.points.shape[0]
            and cluster < inputs.centers.shape[0]
            and approximate - error <= limit
        ):
            value = float(0.0)
            for column in range(wp.static(d)):
                delta = inputs.points[row, column] - inputs.centers[cluster, column]
                value += delta * delta
        return value

    @wp.kernel(enable_backward=False)
    def matmul_assign_kernel(
        x: wp.array2d[float], centers: wp.array2d[float], labels: wp.array[int], distances: wp.array[float]
    ):
        block = wp.tid()
        inputs = DistanceInputs()
        inputs.points = x
        inputs.centers = centers
        points = wp.tile_load(x, shape=(point_tile, width), offset=(block * point_tile, 0))
        point_norm = wp.tile_sum(wp.tile_map(wp.mul, points, points), axis=1)
        point_norm_matrix = wp.tile_broadcast(
            wp.tile_reshape(point_norm, shape=(point_tile, 1)), shape=(point_tile, center_tile)
        )
        row_ids = wp.tile_map(wp.add, wp.tile_arange(point_tile, dtype=int), block * point_tile)
        row_matrix = wp.tile_broadcast(
            wp.tile_reshape(row_ids, shape=(point_tile, 1)), shape=(point_tile, center_tile)
        )
        best = wp.tile_full(shape=(point_tile,), value=wp.inf, dtype=float)
        winners = wp.tile_zeros(shape=(point_tile,), dtype=int)
        for start in range(0, centers.shape[0], center_tile):
            coordinates = wp.tile_load(centers, shape=(center_tile, width), offset=(start, 0))
            center_norm = wp.tile_sum(wp.tile_map(wp.mul, coordinates, coordinates), axis=1)
            center_norm_matrix = wp.tile_broadcast(
                wp.tile_reshape(center_norm, shape=(1, center_tile)), shape=(point_tile, center_tile)
            )
            product = wp.tile_matmul(points, wp.tile_transpose(coordinates), precision="tf32")
            approximate = wp.tile_map(approximate_distance, product, point_norm_matrix, center_norm_matrix)
            upper = wp.tile_map(upper_distance, approximate, point_norm_matrix, center_norm_matrix)
            # The zero-padded center slots must not provide a false upper bound.
            ids = wp.tile_map(wp.add, wp.tile_arange(center_tile, dtype=int), start)
            ids_matrix = wp.tile_broadcast(
                wp.tile_reshape(ids, shape=(1, center_tile)), shape=(point_tile, center_tile)
            )
            upper = wp.tile_map(mask_center_distance, upper, ids_matrix, centers.shape[0])
            limit = wp.tile_map(wp.min, wp.tile_reduce(wp.min, upper, axis=1), best)
            limit_matrix = wp.tile_broadcast(
                wp.tile_reshape(limit, shape=(point_tile, 1)), shape=(point_tile, center_tile)
            )
            squared = wp.tile_map(
                refine_distance,
                approximate,
                point_norm_matrix,
                center_norm_matrix,
                limit_matrix,
                row_matrix,
                ids_matrix,
                inputs,
            )
            minima = wp.tile_reduce(wp.min, squared, axis=1)
            minimum_matrix = wp.tile_broadcast(
                wp.tile_reshape(minima, shape=(point_tile, 1)), shape=(point_tile, center_tile)
            )
            candidates = wp.tile_map(choose_label, squared, minimum_matrix, ids_matrix)
            nearest = wp.tile_reduce(wp.min, candidates, axis=1)
            winners = wp.tile_map(merge_label, minima, best, nearest, winners)
            best = wp.tile_map(wp.min, best, minima)
        wp.tile_store(labels, winners, offset=block * point_tile)
        wp.tile_store(distances, best, offset=block * point_tile)

    return matmul_assign_kernel


@wp.func
def mask_center_distance(distance: float, cluster: int, k: int):
    value = distance
    if cluster >= k:
        value = wp.inf
    return value

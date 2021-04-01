
from sklearn.metrics.pairwise import euclidean_distances as dist


# rewrite to run untill the maximum radius of clusters decreases to a given distance, this number is bounded by twice the optimum. 
def incremental_farthest_search(points, k):
    remaining_points = points[:]
    solution_set = []
    solution_set.append(remaining_points.pop(random.randint(0, len(remaining_points) - 1)))
    for _ in range(k-1):
        distances = [dist(p, solution_set[0]) for p in remaining_points]
        for i, p in enumerate(remaining_points):
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], distance(p, s))
        solution_set.append(remaining_points.pop(distances.index(max(distances))))
    return solution_set

    # Cluster points so that r < h*rho_x

    #define coefficient function  as a function of approximation accuracy P

    #define approximation function G(input)
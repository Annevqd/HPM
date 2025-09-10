import numpy as np
import random
import math
import copy
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import random2

import pandas as pd

def preprocess_dataset(current_file, prev_update_file, delta):
    """
    Prepare the current dataset by adding reverse demand and unmet demands from previous period.
    Assumes both datasets have the same customer order.
    """
    # Load current dataset
    data_current = pd.read_fwf(current_file, skiprows=8, header=None)
    data_current.columns = ["CUST_NO", "XCOORD", "YCOORD", "DEMAND", "READY_TIME", "DUE_DATE", "SERVICE_TIME"]

    # Load updated previous dataset
    data_prev = pd.read_csv(prev_update_file, sep="\t")

    # Compute reverse demand (based on satisfied forward demand from previous period)
    rev_demand_prev = (data_prev["SATISFIED_FWD"] * delta).round().astype(int)

    # Add required columns
    data_current["REV_DEMAND_PREV"] = rev_demand_prev
    data_current["UNSATISFIED_FWD_PREV"] = data_prev["UNSATISFIED_FWD"]
    data_current["UNSATISFIED_REV_PREV"] = data_prev["UNSATISFIED_REV"]

    # Fill depot values with zeros
    data_current.loc[0, ["REV_DEMAND_PREV", "UNSATISFIED_FWD_PREV", "UNSATISFIED_REV_PREV"]] = 0

    # Save enriched dataset for verification
    data_current.to_csv("current_period_enriched.txt", sep="\t", index=False)

    return data_current

delta = 0.14        #https://www-statista-com.tudelft.idm.oclc.org/statistics/1613605/online-returns-e-commerce-europe/
data = preprocess_dataset("rc102.txt", "rc101_update.txt", delta)

print("\n=== Full Current Period Dataset ===")
print(data.head())


# === Sets ===
V = list(range(25))  # 25 vehicles
C = list(range(1, len(data))) # Customers (1 to n), excluding depot (0)
N = list(range(len(data)))  # All nodes including depot
# === Parameters ===
dc = (data["DEMAND"] + data["UNSATISFIED_FWD_PREV"]).tolist()
dc[0] = 0  # Depot demand = 0
dc_rev =(data["REV_DEMAND_PREV"] + data["UNSATISFIED_REV_PREV"]).tolist()
dc_rev[0] = 0  # Depot demand = 0
coords = list(zip(data['XCOORD'], data['YCOORD']))
ai = data['READY_TIME'].tolist()
bi = data['DUE_DATE'].tolist()
si = data['SERVICE_TIME'].tolist()


# Problem parameters
cij = 30
lambda_v = 60
pi = 50
theta = 0.8
mv = 1000
cv = 100
hv = 15
ev = 0.1
cf = 0.5                                                                                        
fv = 0.1
gamma = 0.1
M = 1000000
w1, w2, w3 = 1, 1, 1  # weight for objectives
alpha_w, beta_w, gamma_w = 1, 1, 1  # penalty weights

# === Distance matrix ===
t = np.zeros((len(N), len(N)))
for i in N:
    for j in N:
        t[i][j] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])


# === Initialize weights and scores ===
destruction_weights = [1.0] * 7
repair_weights = [1.0] * 6
destruction_scores = [0] * 7
repair_scores = [0] * 6
destruction_usage = [1] * 7
repair_usage = [1] * 6
alpha = 0.9  # weight update parameter


# === Roulette Wheel Selection ===
def select_operator(weights):
    total = sum(weights)
    r = random.uniform(0, total)
    cumulative = 0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return i
    return len(weights) - 1  # fallback

# === Update Weights ===
def update_weights(weights, scores, usage):
    for i in range(len(weights)):
        weights[i] = alpha * weights[i] + (1 - alpha) * (scores[i] / usage[i] if usage[i] else 0)
        scores[i] = 0
        usage[i] = 1


# === Destruction Operators ===

# 1. Random Destroy Operator
def random_destroy(routes, fraction=0.8):
    all_customers = [(v, cust) for v in range(len(routes)) for cust in routes[v] if cust != 0]
    num_remove = max(1, int(len(all_customers) * fraction))
    to_remove = random.sample(all_customers, num_remove)
    removed = []
    for v, cust in to_remove:
        if cust in routes[v]:
            routes[v].remove(cust)
            removed.append(cust)
    return removed, routes

# 2. Similarity Destroy Operator
def similarity_destroy(routes, coords, fraction=0.6):
    total_customers = sum(1 for r in routes for c in r if c != 0)
    q = max(1, int(fraction * total_customers))
    # Normalize distances
    max_dist = max(math.dist(a, b) for a in coords for b in coords if a != b)

    def similarity(i, j):
        dist = math.dist(coords[i], coords[j]) / max_dist  # normalized c_ij
        same_route = any(i in r and j in r for r in routes)
        Vij = 0 if same_route else 1
        return 1 / (dist + Vij)

    all_customers = [cust for r in routes for cust in r if cust != 0]
    seed = random.choice(all_customers)
    removed = [seed]

    while len(removed) < q:
        curr = random.choice(removed)
        candidates = [c for c in all_customers if c not in removed]
        if not candidates:
            break

        candidates.sort(key=lambda j: -similarity(curr, j))
        rand_index = int(random.random() * len(candidates))
        selected = candidates[rand_index]
        removed.append(selected)

    for v in range(len(routes)):
        routes[v] = [c for c in routes[v] if c not in removed]

    return removed, routes

# 3. Maximum Saving Cost Destroy Operator
def max_saving_destroy(routes, coords, fraction=0.6):
    total_customers = sum(1 for r in routes for c in r if c != 0)
    q = max(1, int(fraction * total_customers))
    savings = []
    for v, route in enumerate(routes):
        for i in range(1, len(route) - 1):
            prev, curr, next_ = route[i-1], route[i], route[i+1]
            saving = t[prev][curr] + t[curr][next_] - t[prev][next_]
            savings.append((saving, v, curr))
    savings.sort(reverse=True)
    removed = [cust for _, v, cust in savings[:q]]
    for _, v, cust in savings[:q]:
        routes[v].remove(cust)
    return removed, routes

# 4. Destroying Vehicle Destroy Operator
def destroying_vehicle_destroy(routes):
    max_cost = -1
    vehicle_to_remove = -1

    for v, route in enumerate(routes):
        if not route or len(route) < 2:
            continue

        # Path length
        path_length = sum(t[route[i]][route[i+1]] for i in range(len(route) - 1))

        # Average waiting time
        time = 0
        wait_total = 0
        visits = 0

        for i in range(1, len(route)):
            prev = route[i - 1]
            curr = route[i]
            travel = t[prev][curr]
            time += travel

            if curr != 0:  # Ignore depot
                wait = max(0, ai[curr] - time)
                wait_total += wait
                time += si[curr]
                visits += 1

        avg_wait = wait_total / visits if visits > 0 else 0

        # Weighted cost metric
        cost = 0.7 * path_length + 0.3 * avg_wait

        if cost > max_cost:
            max_cost = cost
            vehicle_to_remove = v

    # Remove all customers from that vehicle's route
    removed = [cust for cust in routes[vehicle_to_remove] if cust != 0]
    routes[vehicle_to_remove] = []

    return removed, routes


# 5. Maximum Waiting Time Destroy Operator 
def max_waiting_time_destroy(routes, ai, bi, si):
    def calc_wait(route):
        time = 0
        wait_total = 0
        for i in range(1, len(route)):
            prev = route[i - 1]
            curr = route[i]
            travel = t[prev][curr]
            time += travel

            # Actual waiting if early
            if curr != 0:
                wait = max(0, ai[curr] - time)

                # Penalty if late
                late_penalty = max(0, time - bi[curr]) * 10

                wait_total += wait + late_penalty

                time += si[curr]
        return wait_total

    vehicle_to_remove = max(range(len(routes)), key=lambda v: calc_wait(routes[v]))
    removed = [cust for cust in routes[vehicle_to_remove] if cust != 0]
    routes[vehicle_to_remove] = []
    return removed, routes

# 6. 2-Opt Search Algorithm

from py2opt.routefinder import RouteFinder

def destroy_2opt(routes, dist_matrix, iterations=10):
    """
    Applies 2-opt on each non-empty route using py2opt and returns the new solution.
    No customers are removed, so `removed_customers` is an empty list.
    """
    new_solution = []
    removed_customers = []  # 2-opt does not remove, just reorders within a route

    for route in routes:
        if len(route) <= 3:  # No improvement possible for short routes
            new_solution.append(route)
            continue

        # Ensure route starts and ends at depot
        if route[0] != 0:
            route = [0] + route
        if route[-1] != 0:
            route.append(0)

        # Skip if route has no customers
        customer_count = sum(1 for node in route if node != 0)
        if customer_count < 2:
            new_solution.append(route)
            continue

        # Convert to string labels for py2opt
        route_labels = [str(c) for c in route]
        submatrix = [[dist_matrix[a][b] for b in route] for a in route]

        # Run 2-opt
        rf = RouteFinder(submatrix, route_labels, iterations=iterations)
        _, best_route = rf.solve()

        # Convert back to integers
        optimized_route = [int(c) for c in best_route]

        # Double-check depot structure
        if optimized_route[0] != 0:
            optimized_route = [0] + optimized_route
        if optimized_route[-1] != 0:
            optimized_route.append(0)

        new_solution.append(optimized_route)

    return removed_customers, new_solution


def shaw_destroy_pdptw(routes, coords, ai, bi, si, dc, fraction=0.6, p=100, 
                       w_dist=1.0, w_time=1.0, w_cap=1.0, w_veh=1.0):
    """
    Shaw removal heuristic for PDPTW with distance, time, capacity, and vehicle compatibility.
    
    routes: list of vehicle routes (list of customer indices)
    coords: coordinates of each node
    ai, bi: time windows
    si: service times
    dc: demands (positive for pickup, negative for delivery)
    vehicle_feasibility: list of sets, where vehicle_feasibility[i] = vehicles that can serve customer i
    q: number of customers to remove
    p: determinism parameter (>=1)
    weights: w_dist, w_time, w_cap, w_veh for relatedness terms
    """
    
    total_customers = sum(1 for r in routes for c in r if c != 0)
    q = max(1, int(fraction * total_customers))

    # --- Flatten customers from all routes ---
    all_customers = [cust for r in routes for cust in r if cust != 0]
    if len(all_customers) <= q:
        return all_customers, [[] for _ in routes]  # remove all
    
    # --- Precompute normalization factors ---
    max_dist = max(t[i][j] for i in all_customers for j in all_customers if i != j)
    max_time_diff = max(abs(ai[i] - ai[j]) + abs(bi[i] - bi[j]) 
                        for i in all_customers for j in all_customers if i != j)
    max_demand_diff = max(abs(dc[i] - dc[j]) for i in all_customers for j in all_customers if i != j)
    
    # --- Relatedness function ---
    def relatedness(i, j):
        dist_term = (t[i][j] / max_dist) if max_dist > 0 else 0
        time_term = ((abs(ai[i] - ai[j]) + abs(bi[i] - bi[j])) / max_time_diff) if max_time_diff > 0 else 0
        cap_term = (abs(dc[i] - dc[j]) / max_demand_diff) if max_demand_diff > 0 else 0
        
        return w_dist*dist_term + w_time*time_term + w_cap*cap_term
    
    # --- Start removal ---
    removed = []
    seed = random.choice(all_customers)
    removed.append(seed)
    
    while len(removed) < q:
        # Build candidate list (not yet removed)
        candidates = [c for c in all_customers if c not in removed]
        
        # Find relatedness to already removed set (take min relatedness to any removed)
        rel_scores = []
        for c in candidates:
            rel_min = min(relatedness(c, r) for r in removed)
            rel_scores.append((rel_min, c))
        
        # Sort by relatedness (lowest = most related)
        rel_scores.sort(key=lambda x: x[0])
        
        # Probabilistic selection using determinism parameter p
        idx = int(random.random() ** p * len(rel_scores))
        selected = rel_scores[idx][1]
        
        removed.append(selected)
    
    # --- Remove from routes ---
    new_routes = [[c for c in r if c not in removed] for r in routes]
    
    return removed, new_routes



destruction_ops = [
    lambda r: random_destroy(r, fraction=0.8),
    lambda r: similarity_destroy(r, coords, fraction=0.6),
    lambda r: max_saving_destroy(r, coords, fraction=0.6),
    lambda r: max_waiting_time_destroy(r, ai, bi, si),
    lambda r: destroying_vehicle_destroy(r),
    lambda r: destroy_2opt(r, t),
    lambda r: shaw_destroy_pdptw(r, coords, ai, bi, si, dc, fraction=0.6, p=100, 
                       w_dist=1.0, w_time=1.0, w_cap=1.0, w_veh=1.0),
]


# === Repair Operators ===

# Delta cost evaluation function
def delta_insertion_cost(route, i, cust):
    prev = route[i - 1]
    next_ = route[i] if i < len(route) else 0
    delta = t[prev][cust] + t[cust][next_] - t[prev][next_]
    return delta

# 1. Global Optimal Repair Operator
def global_optimal_repair(routes, removed):
    for cust in removed:
        best_cost = float('inf')
        best_pos = None
        for v, route in enumerate(routes):
            for i in range(1, len(route)):
                if not is_time_window_feasible(route, i, cust, ai, bi, si, t):
                    continue
                if not is_capacity_feasible(route, i, cust, dc, dc_rev, lambda_v):
                    continue
                delta = delta_insertion_cost(route, i, cust)
                if delta < best_cost:
                    best_cost = delta
                    best_pos = (v, i)
        if best_pos:
            v, i = best_pos
            routes[v].insert(i, cust)
        else:
            # Start a new route if no feasible position found
            routes.append([0, cust, 0])
    return routes

# 2. Minimum Insertion Cost Repair Operator
def min_insertion_cost_repair(routes, removed):
    pending = removed[:]
    while pending:
        best_costs = []

        for cust in pending:
            best_delta = float("inf")
            best_wait = float("inf")
            best_pos = None

            for v, route in enumerate(routes):
                for i in range(1, len(route)):
                    if not is_time_window_feasible(route, i, cust, ai, bi, si, t):
                        continue
                    if not is_capacity_feasible(route, i, cust, dc, dc_rev, lambda_v):
                        continue

                    # Delta cost (local evaluation)
                    prev, next_ = route[i - 1], route[i]
                    delta_cost = t[prev][cust] + t[cust][next_] - t[prev][next_]

                    # Estimate waiting time after insertion
                    arrival = 0
                    for j in range(1, i):
                        arrival += t[route[j - 1]][route[j]] + si[route[j - 1]]
                    arrival += t[route[i - 1]][cust]
                    wait_time = max(0, ai[cust] - arrival)

                    if (delta_cost < best_delta) or (delta_cost == best_delta and wait_time < best_wait):
                        best_delta = delta_cost
                        best_wait = wait_time
                        best_pos = (v, i)

            best_costs.append((best_delta, cust, best_pos))

        # Choose customer with largest minimum insertion cost
        best_costs.sort(reverse=True)
        _, selected_cust, selected_pos = best_costs[0]

        if selected_pos:
            v, i = selected_pos
            routes[v].insert(i, selected_cust)
        else:
            # No feasible insertion thencreate a new route
            routes.append([0, selected_cust, 0])

        pending.remove(selected_cust)

    return routes

# 3. Random K Repair Operator
def random_k_repair(routes, removed, ai, bi, si, t, dc, dc_rev, lambda_v, current_iter):
    
    k_dynamic = 5 + current_iter // 500

    for cust in removed:
        candidates = []

        for v, route in enumerate(routes):
            for i in range(1, len(route)):  # Skip depot at 0
                if not is_time_window_feasible(route, i, cust, ai, bi, si, t):
                    continue
                if not is_capacity_feasible(route, i, cust, dc, dc_rev, lambda_v):
                    continue

                prev, next_ = route[i - 1], route[i]
                delta_cost = t[prev][cust] + t[cust][next_] - t[prev][next_]

                candidates.append((delta_cost, v, i))

        if candidates:
            candidates.sort()
            chosen = random.choice(candidates[:min(k_dynamic, len(candidates))])
            _, v, i = chosen
            routes[v].insert(i, cust)
        else:
            routes.append([0, cust, 0])  # New route if no insertion possible

    return routes


# 4. Regret Criterion Repair Operator
def regret_repair(routes, removed, ai, bi, si, t, dc, dc_rev, lambda_v, k=10):

    while removed:
        regrets = []

        for cust in removed:
            candidate_positions = []

            for v, route in enumerate(routes):
                for i in range(1, len(route)):
                    if not is_time_window_feasible(route, i, cust, ai, bi, si, t):
                        continue
                    if not is_capacity_feasible(route, i, cust, dc, dc_rev, lambda_v):
                        continue
                    prev, next_ = route[i - 1], route[i]
                    delta = t[prev][cust] + t[cust][next_] - t[prev][next_]
                    candidate_positions.append((delta, v, i))

            if len(candidate_positions) >= k:
                candidate_positions.sort()
                regret = sum(candidate_positions[j][0] - candidate_positions[0][0] for j in range(1, k))
                regrets.append((regret, cust, candidate_positions[0]))

        if regrets:
            regrets.sort(reverse=True)  # Max regret first
            _, chosen_cust, (best_delta, v, i) = regrets[0]
            routes[v].insert(i, chosen_cust)
            removed.remove(chosen_cust)
        else:
            # No feasible insertion for any customer, create new route for one
            chosen_cust = removed.pop(0)
            routes.append([0, chosen_cust, 0])

    return routes


# 5. Minimum Waiting Time Repair Operator
def min_waiting_time_repair(routes, removed, ai, bi, si, t, dc, dc_rev, lambda_v):
    for cust in removed:
        best_pos = None
        min_wait = float('inf')

        for v, route in enumerate(routes):
            time = 0  # cumulative time along the route

            for i in range(1, len(route)):
                prev = route[i - 1]
                departure_prev = time + si[prev]
                travel = t[prev][cust]
                arrival = max(departure_prev + travel, ai[cust])
                wait = arrival - departure_prev - travel

                if not is_time_window_feasible(route, i, cust, ai, bi, si, t):
                    continue
                if not is_capacity_feasible(route, i, cust, dc, dc_rev, lambda_v):
                    continue

                if wait < min_wait:
                    min_wait = wait
                    best_pos = (v, i)

                # update time to next node
                next_node = route[i]
                time = departure_prev + t[prev][next_node]

        if best_pos:
            v, i = best_pos
            routes[v].insert(i, cust)
        else:
            routes.append([0, cust, 0])  # New route for isolated customer

    return routes



def lookahead_insertion_repair(routes, removed):
    for cust in removed:
        best_score = float("inf")
        best_pos = None
        for v, route in enumerate(routes):
            for i in range(1, len(route)):
                if not is_time_window_feasible(route, i, cust, ai, bi, si, t):
                    continue
                if not is_capacity_feasible(route, i, cust, dc, dc_rev, lambda_v):
                    continue
                delta1 = delta_insertion_cost(route, i, cust)
                # Lookahead: assume a random next removal could be placed after
                delta2 = 0
                if i < len(route) - 1:
                    delta2 = delta_insertion_cost(route, i + 1, cust)
                score = delta1 + 0.5 * delta2
                if score < best_score:
                    best_score = score
                    best_pos = (v, i)
        if best_pos:
            v, i = best_pos
            routes[v].insert(i, cust)
        else:
            routes.append([0, cust, 0])
    return routes


repair_ops = [
    lambda r, rem: global_optimal_repair(r, rem),
    lambda r, rem: min_insertion_cost_repair(r, rem),
    lambda r, rem: regret_repair(r, rem, ai, bi, si, t, dc, dc_rev, lambda_v, k=10),
    lambda r, rem: random_k_repair(r, rem, ai, bi, si, t, dc, dc_rev, lambda_v, current_iter=0),
    lambda r, rem: min_waiting_time_repair(r, rem, ai, bi, si, t, dc, dc_rev, lambda_v),
    lambda r, rem: lookahead_insertion_repair(r, rem)
]


# === 1. Nearest Neighbor Heuristic ===
def nearest_neighbor_initial_solution():
    unvisited = set(C)
    routes = [[] for _ in V]
    v = 0

    while unvisited and v < len(V):
        route = [0]
        load = 0
        curr = 0
        while True:
            nearest = None
            nearest_dist = float('inf')
            for cust in unvisited:
                if dc[cust] + load <= lambda_v and t[curr][cust] < nearest_dist:
                    nearest = cust
                    nearest_dist = t[curr][cust]
            if nearest is None:
                break
            route.append(nearest)
            load += dc[nearest]
            curr = nearest
            unvisited.remove(nearest)
        route.append(0)
        routes[v] = route
        v += 1
    return routes
initial_routes = nearest_neighbor_initial_solution()

import json
with open("routes_nns.json", "w") as f:
    json.dump(initial_routes, f)

    # Preview output
for v, route in enumerate(initial_routes):
    if route:
        print(f"Vehicle {v}: {route}")


def _simulate_route_with_start_load(route, start_load, unmet_delivery, unmet_return,
                                    ai, bi, si, t, lambda_v, log_profile=False):
    """
    Simulate one route with a given start_load.
    Returns:
        feasible, arrival, waiting_total, Fc_inc, Rc_inc, end_load, tw_penalty,
        profile (if log_profile=True)
    """
    n = max(max(route), 0) + 1
    Fc_inc = [0.0] * n
    Rc_inc = [0.0] * n

    load = start_load
    arrival = 0.0
    waiting_total = 0.0
    tw_penalty = 0.0
    feasible = True

    profile_rows = [] if log_profile else None
    cum_reverse = 0.0
    remaining_forward_total = sum(unmet_delivery.get(i, 0) for i in route if i != 0)

    for k in range(len(route) - 1):
        i = route[k]
        j = route[k + 1]

        if i != 0:
            # Arrive at i
            if arrival < ai[i]:
                waiting_total += (ai[i] - arrival)
                arrival = ai[i]
            arrival += si[i]

            # Deliver
            deliverable = min(unmet_delivery[i], load)
            unmet_delivery[i] -= deliverable
            load -= deliverable
            Fc_inc[i] += round(deliverable, 0)

            # Pickup
            pickup_cap = lambda_v - load
            pickup = min(unmet_return[i], pickup_cap)
            unmet_return[i] -= pickup
            load += pickup
            Rc_inc[i] += round(pickup,0)

            cum_reverse += pickup
            remaining_forward_total -= deliverable

            if log_profile:
                profile_rows.append({
                    "visit_idx": len(profile_rows) + 1,
                    "node": i,
                    "served_forward": deliverable,
                    "cumulative_reverse": cum_reverse,
                    "net_load_after": load
                })

            # Capacity check
            if load < -1e-9 or load > lambda_v + 1e-9:
                feasible = False

        # Travel to next
        arrival += t[i][j]

        # TW penalty
        if j != 0:
            if arrival > bi[j] + 1e-9:
                tw_penalty += (arrival - bi[j])

    if log_profile:
        return feasible, arrival, waiting_total, Fc_inc, Rc_inc, load, tw_penalty, profile_rows
    else:
        return feasible, arrival, waiting_total, Fc_inc, Rc_inc, load, tw_penalty



def _choose_feasible_start_load(route, unmet_delivery, unmet_return,
                                ai, bi, si, t, lambda_v):
    """
    Find the largest start load in [0, lambda_v] that keeps the route's load within [0, lambda_v]
    under greedy 'deliver then pick up'. We ignore TW here; we choose L0 for capacity feasibility.
    """
    lo, hi = 0.0, float(lambda_v)

    # Quick upper bound: no need to load more forward than still unmet along this route
    route_demand = sum(unmet_delivery[i] for i in route if i != 0)
    hi = min(hi, route_demand)

    # Binary search
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        # Simulate with copies (don't mutate the originals in the test)
        ud = unmet_delivery.copy()
        ur = unmet_return.copy()
        feasible, *_ = _simulate_route_with_start_load(route, mid, ud, ur, ai, bi, si, t, lambda_v)
        if feasible:
            lo = mid
        else:
            hi = mid
    return lo  # maximal feasible (up to tolerance)


# === a Feasibility Checker Function ===

# Helper function for capacity check
def is_capacity_feasible(route, insert_pos, cust, dc, dc_rev, lambda_v):
    # Create a tentative new route with cust inserted at the position
    new_route = route[:insert_pos] + [cust] + route[insert_pos:]

    load = 0  # initial load is 0
    for node in new_route:
        if node == 0:
            continue  # depot, skip
        # Deliver forward items
        deliver = dc[node]
        if load >= deliver:
            load -= deliver
        else:
            deliver = load  # deliver what we can
            load = 0
        # Pick up reverse items (adds to load)
        pickup = dc_rev[node]
        load += pickup
        # Check capacity after pickup
        if load > lambda_v:
            return False
    return True


def is_time_window_feasible(route, insert_pos, cust, ai, bi, si, t):
    prev = route[insert_pos - 1]
    next_ = route[insert_pos] if insert_pos < len(route) else 0

    # Estimate earliest possible arrival at 'cust'
    est = max(ai[cust], ai[prev] + si[prev] + t[prev][cust])
    lst = bi[cust]

    # Check if insertion affects the next node's feasibility
    if next_ != 0:
        latest_next = bi[next_] - t[cust][next_] - si[cust]
        if est > latest_next:
            return False

    return est <= lst

# === Evaluation F(X) + penalty(X) with Forward/Backward Slack for Time Windows ===
def evaluate_with_penalty_milp_logic(routes, dc, dc_rev, ai, bi, si, t, lambda_v, mv,
                                     hv, cv, cf, fv, cij, ev, gamma, pi, theta, w1, w2, w3, C,
                                     log_profiles=False):
    """
    Evaluation function matching MILP constraint logic exactly.
    """
    # Track total components
    Feco = 0.0
    Fenv = 0.0
    Fsoc = 0.0

        # Initialize arrays for dataset output
    Fc_total = [0.0] * len(dc)  # satisfied forward
    Rc_total = [0.0] * len(dc)  # satisfied reverse
    Qd_vals = [0.0] * len(dc)   # unmet forward
    Qr_vals = [0.0] * len(dc)   # unmet reverse


    # Track unmet demand
    unmet_delivery = {i: dc[i] for i in C}
    unmet_return = {i: dc_rev[i] for i in C}

    vehicles_used = 0
    total_route_time = 0.0
    tw_violations_count = 0
    capacity_violations_count = 0
    mv_violations_count = 0

    # Penalty tracking
    penalty_tw_total = 0.0
    penalty_capacity_total = 0.0
    penalty_mv_total = 0.0

    vehicle_profiles = {}  # new: stores per-vehicle load profiles if log_profiles=True
    initial_loads = {}

    for v_idx, route in enumerate(routes):
        # Ignore empty or depot-only routes
        customer_nodes = [node for node in route if node != 0]
        if not customer_nodes:
            continue

        vehicles_used += 1

        # === Arrival time & waiting logic ===
        arrival = 0.0
        waiting_total = 0.0

        # 1) choose a feasible initial load L0 in [0, lambda_v]
        L0 = _choose_feasible_start_load(route, unmet_delivery, unmet_return,
                                 ai, bi, si, t, lambda_v)
        initial_loads[v_idx] = L0
        
        # 2) now simulate once to compute Fc/Rc and times with this L0
        ud = unmet_delivery  # mutate the shared dictionaries (this is the real run)
        ur = unmet_return
        if log_profiles:
            feasible, arrival_end, waiting_total, Fc_inc, Rc_inc, end_load, tw_penalty, profile_rows = _simulate_route_with_start_load(route, L0, ud, ur, ai, bi, si, t, lambda_v, log_profile=True)
            vehicle_profiles[v_idx] = profile_rows
        else:
            feasible, arrival_end, waiting_total, Fc_inc, Rc_inc, end_load, tw_penalty = _simulate_route_with_start_load(route, L0, ud, ur, ai, bi, si, t, lambda_v)


                # ===== NEW: log profile if requested =====
        if log_profiles:
            profile_rows = []
            load = L0
            cum_rev = 0.0
            remaining_fwd = sum(ud.get(i, 0) for i in customer_nodes)

            prev = route[0]
            arrival = 0.0
            for k in range(1, len(route)):
                curr = route[k]
                arrival += t[prev][curr]
                if curr != 0:
                    wait = max(0, ai[curr] - arrival)
                    start_service = arrival + wait
                    end_service = start_service + si[curr]

                    deliverable = Fc_inc[curr]
                    pickup = Rc_inc[curr]
                    load -= deliverable
                    load += pickup
                    cum_rev += pickup
                    remaining_fwd -= deliverable

                    profile_rows.append({
                        "visit_idx": len(profile_rows) + 1,
                        "node": curr,
                        "served_forward": deliverable,
                        "cumulative_reverse": cum_rev,
                        "net_load_after": load
                    })
                    arrival = end_service
                prev = curr
            
            vehicle_profiles[v_idx] = profile_rows

        # record per-customer satisfied amounts (MILP-consistent, no negatives)
        for i in range(len(Fc_inc)):
            if i < len(Fc_total):
                Fc_total[i] += max(0.0, Fc_inc[i])
                Rc_total[i] += max(0.0, Rc_inc[i])

        # time-window penalty (scaled like before; you can keep your factor 1000)
        penalty_tw_total += 1000.0 * tw_penalty

        # 3) build beta_v = travel + service + waiting
        travel_time = sum(t[route[k]][route[k + 1]] for k in range(len(route) - 1))
        service_time = sum(si[i] for i in customer_nodes)
        beta_v = travel_time + service_time
        total_route_time += beta_v

        # 4) max-route-time penalty (same as before)
        if beta_v > mv + 1e-6:
            penalty_mv_total += 10000.0 * (beta_v - mv)
            mv_violations_count += 1


        # Economic cost components
        vehicle_fixed_cost = cv
        driver_wage = hv * beta_v
        transport_cost = sum(t[route[i]][route[i+1]] * (cf * fv + cij) for i in range(len(route)-1))
        Feco += vehicle_fixed_cost + driver_wage + transport_cost

        # Environmental cost
        env_cost = sum(ev * t[route[i]][route[i+1]] * gamma for i in range(len(route)-1))
        Fenv += env_cost

        # Social cost
        Fsoc += driver_wage

    # Penalty for unmet demands
    total_unmet_forward = sum(unmet_delivery.values())
    total_unmet_return = sum(unmet_return.values())
    penalty_unmet_delivery = pi * total_unmet_forward
    penalty_unmet_return = pi * (1 - theta) * total_unmet_return
    penalty_unmet = penalty_unmet_delivery + penalty_unmet_return
    Feco += penalty_unmet

    for i in C:
        Qd_vals[i] = round(unmet_delivery[i],0)
        Qr_vals[i] = round(unmet_return[i],0)


    violation_penalty = penalty_tw_total + penalty_capacity_total + penalty_mv_total

    # Final weighted objective
    total_cost = w1 * Feco + w2 * Fenv + w3 * Fsoc + violation_penalty

    penalty_breakdown = {
        'time_window_penalty': penalty_tw_total,
        'capacity_penalty': penalty_capacity_total,
        'max_time_penalty': penalty_mv_total,
        'unmet_delivery_penalty': penalty_unmet_delivery,
        'unmet_return_penalty': penalty_unmet_return,
        'total_violation_penalty': violation_penalty,
        'vehicles_used': vehicles_used,
        'total_route_time': total_route_time,
        'tw_violations_count': tw_violations_count,
        'capacity_violations_count': capacity_violations_count,
        'mv_violations_count': mv_violations_count,
        'total_unmet_forward': total_unmet_forward,
        'total_unmet_return': total_unmet_return
    }

    return (Feco, Fenv, Fsoc, total_cost, violation_penalty, penalty_breakdown,
            Fc_total, Rc_total, Qd_vals, Qr_vals,
            vehicle_profiles if log_profiles else None,
            initial_loads)


def ialns_sa(initial_solution, evaluate_func, destruction_ops, repair_ops,
             T0=100, Tmin=0.01, cooling_rate=0.995, max_iter=3000,
             pool_max_size=10, reintro_interval=250):

    current = copy.deepcopy(initial_solution)
    best = copy.deepcopy(initial_solution)

    Feco_best, Fenv_best, Fsoc_best, best_cost, best_penalty, best_breakdown, final_Fc_total, final_Rc_total, final_Qd_vals, final_Qr_vals, vehicle_profiles, initial_loads = evaluate_func(best, dc, dc_rev, ai, bi, si, t, lambda_v, mv,
                                     hv, cv, cf, fv, cij, ev, gamma, pi, theta, w1, w2, w3, C,
                                     log_profiles=True)
    current_cost = best_cost

    T = T0
    iteration = 0

    d_weights = [1.0] * len(destruction_ops)
    r_weights = [1.0] * len(repair_ops)

    feasible_pool = []

    print(f"Initial solution cost: {best_cost:.2f}")

    while T > Tmin and iteration < max_iter:
        # === 1. Select operators using roulette wheel ===
        d_idx = select_operator(d_weights)
        r_idx = select_operator(r_weights)

        # === 2. Track usage of selected operators ===
        destruction_usage[d_idx] += 1
        repair_usage[r_idx] += 1

        # === 3. Apply destruction and repair ===
        removed, destroyed = destruction_ops[d_idx](copy.deepcopy(current))

        try:
            repaired = repair_ops[r_idx](copy.deepcopy(destroyed), removed)
        except TypeError:
            # For operators needing 'iteration' param
            repaired = repair_ops[r_idx](copy.deepcopy(destroyed), removed, iteration)

        # === 4. Evaluate solution ===
        Feco, Fenv, Fsoc, new_cost, penalty, breakdown, final_Fc, final_Rc, final_Qd, final_Qr, vehicle_profiles, initial_loads = evaluate_func(repaired, dc, dc_rev, ai, bi, si, t, lambda_v, mv,
                                     hv, cv, cf, fv, cij, ev, gamma, pi, theta, w1, w2, w3, C,
                                     log_profiles=True)

        # === 5. Accept or reject ===
        if penalty == 0:
            feasible_pool.append(copy.deepcopy(repaired))
            if len(feasible_pool) > pool_max_size:
                feasible_pool.pop(0)

            if new_cost < best_cost:
                best = copy.deepcopy(repaired)
                best_cost = new_cost
                Feco_best, Fenv_best, Fsoc_best, best_penalty = Feco, Fenv, Fsoc, penalty
                best_breakdown = breakdown
                print(f"New best at iteration {iteration}: {best_cost:.2f}")

        delta = new_cost - current_cost
        if penalty == 0 and (delta < 0 or random.random() < math.exp(-delta / T)):
            current = copy.deepcopy(repaired)
            current_cost = new_cost

        # === 6. Diversification: Reintroduce from pool ===
        if iteration > 0 and iteration % reintro_interval == 0 and feasible_pool:
            current = copy.deepcopy(random.choice(feasible_pool))
            _, _, _, current_cost, _, _, _, _, _, _, _, _ = evaluate_func(current, dc, dc_rev, ai, bi, si, t, lambda_v, mv,
                                     hv, cv, cf, fv, cij, ev, gamma, pi, theta, w1, w2, w3, C,
                                     log_profiles=True)

        # === 7. Print progress every 100 iterations or if penalty occurred ===
        if iteration % 100 == 0 or penalty > 0:
            print(f"Iteration {iteration+1}: Cost = {new_cost:.2f}, Penalty = {penalty:.2f}")
            for v, r in enumerate(repaired):
                if r:
                    print(f"  Vehicle {v}: {r}")

        # === 8. Cooling schedule ===
        T *= cooling_rate
        iteration += 1

    return best, best_cost, Feco_best, Fenv_best, Fsoc_best, best_penalty, best_breakdown, final_Fc_total, final_Rc_total, final_Qd_vals, final_Qr_vals, vehicle_profiles, initial_loads


def print_detailed_solution(routes, cost_breakdown, penalty_breakdown, title="Solution"):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Routes
    print("Routes:")
    active_routes = 0
    total_customers_served = 0
    for v, route in enumerate(routes):
        if route and len(route) > 2:
            customers = [node for node in route if node != 0]
            total_customers_served += len(customers)
            active_routes += 1
            print(f"  Vehicle {v}: {route}")
    
    print(f"\nRoute Statistics:")
    print(f"  Active vehicles: {active_routes}")
    print(f"  Total customers served: {total_customers_served}")
    print(f"  Total vehicles available: {len(V)}")
    
    # Cost Breakdown
    Feco, Fenv, Fsoc, total_cost = cost_breakdown
    print(f"\nCost Breakdown:")
    print(f"  Economic Cost (Feco):     {Feco:>10.2f}")
    print(f"  Environmental Cost (Fenv): {Fenv:>10.2f}")
    print(f"  Social Cost (Fsoc):       {Fsoc:>10.2f}")
    print(f"  {'─'*35}")
    print(f"  Total Objective Cost:     {total_cost:>10.2f}")
    
    # Detailed Penalty Breakdown
    if penalty_breakdown:
        print(f"\nPenalty Breakdown:")
        print(f"  Time Window Violations:")
        print(f"    Penalty:                {penalty_breakdown['time_window_penalty']:>10.2f}")
        print(f"    Violations count:       {penalty_breakdown['tw_violations_count']:>10.0f}")
        
        print(f"  Capacity Violations:")
        print(f"    Penalty:                {penalty_breakdown['capacity_penalty']:>10.2f}")
        print(f"    Violations count:       {penalty_breakdown['capacity_violations_count']:>10.0f}")
        
        print(f"  Maximum Time Violations:")
        print(f"    Penalty:                {penalty_breakdown['max_time_penalty']:>10.2f}")
        print(f"    Violations count:       {penalty_breakdown['mv_violations_count']:>10.0f}")
        
        print(f"  Unmet Demand Penalties:")
        print(f"    Forward demand penalty: {penalty_breakdown['unmet_delivery_penalty']:>10.2f}")
        print(f"    Return demand penalty:  {penalty_breakdown['unmet_return_penalty']:>10.2f}")
        print(f"    Total unmet forward:    {penalty_breakdown['total_unmet_forward']:>10.1f}")
        print(f"    Total unmet return:     {penalty_breakdown['total_unmet_return']:>10.1f}")
        
        print(f"  {'─'*35}")
        print(f"  Total Constraint Penalty: {penalty_breakdown['total_violation_penalty']:>10.2f}")
        
        print(f"\nOperational Statistics:")
        print(f"  Vehicles used:            {penalty_breakdown['vehicles_used']:>10.0f}")
        print(f"  Total route time:         {penalty_breakdown['total_route_time']:>10.2f}")
        print(f"  Average time per vehicle: {penalty_breakdown['total_route_time']/max(1,penalty_breakdown['vehicles_used']):>10.2f}")
        
        # Solution feasibility status
        is_feasible = penalty_breakdown['total_violation_penalty'] == 0
        feasibility_status = "FEASIBLE" if is_feasible else "INFEASIBLE"
        print(f"  Solution Status:          {feasibility_status}")
        
        if not is_feasible:
            print(f"\n  Constraint Violations Summary:")
            if penalty_breakdown['tw_violations_count'] > 0:
                print(f"    - {penalty_breakdown['tw_violations_count']} time window violations")
            if penalty_breakdown['capacity_violations_count'] > 0:
                print(f"    - {penalty_breakdown['capacity_violations_count']} capacity violations")
            if penalty_breakdown['mv_violations_count'] > 0:
                print(f"    - {penalty_breakdown['mv_violations_count']} maximum time violations")
            if penalty_breakdown['total_unmet_forward'] > 0:
                print(f"    - {penalty_breakdown['total_unmet_forward']:.1f} units of unmet forward demand")
            if penalty_breakdown['total_unmet_return'] > 0:
                print(f"    - {penalty_breakdown['total_unmet_return']:.1f} units of unmet return demand")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    print("Generating initial solution...")
    initial_routes = nearest_neighbor_initial_solution()

    print("Initial Solution:")
    for v, route in enumerate(initial_routes):
        if route and len(route) > 2:
            print(f"Vehicle {v}: {route}")

# Run initial solution evaluation
Feco, Fenv, Fsoc, total_cost, violation_penalty, penalty_breakdown, Fc_total, Rc_total, Qd_vals, Qr_vals, vehicle_profiles, initial_loads = evaluate_with_penalty_milp_logic(initial_routes, dc, dc_rev, ai, bi, si, t, lambda_v, mv,
                                     hv, cv, cf, fv, cij, ev, gamma, pi, theta, w1, w2, w3, C,
                                     log_profiles=True)
print_detailed_solution(initial_routes, (Feco, Fenv, Fsoc, total_cost), penalty_breakdown, "Initial Solution (Nearest Neighbor)")

# Run IALNS-SA
print("\nRunning IALNS-SA...")
final_routes, final_cost, final_Feco, final_Fenv, final_Fsoc, final_penalty, final_breakdown, final_Fc, final_Rc, final_Qd, final_Qr, vehicle_profiles, initial_loads  = ialns_sa(
    initial_routes, evaluate_with_penalty_milp_logic, destruction_ops, repair_ops
)

print_detailed_solution(final_routes, (final_Feco, final_Fenv, final_Fsoc, final_cost), final_breakdown, "Final Solution (IALNS-SA)")
    



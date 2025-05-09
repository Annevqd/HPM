#===============================================================================================
#HPM
#===============================================================================================
 
from gurobipy import *
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt



# === Load dataset ===

# Load only the CUSTOMER section of the text file
data = pd.read_fwf("rc104.txt", skiprows=8, header=None)
data.columns = ["CUST_NO", "XCOORD", "YCOORD", "DEMAND", "READY_TIME", "DUE_DATE", "SERVICE_TIME"]


# Load data from previous period
data_pre = pd.read_fwf("rc103.txt", skiprows=8, header=None)
data_pre.columns = ["CUST_NO", "XCOORD", "YCOORD", "DEMAND", "READY_TIME", "DUE_DATE", "SERVICE_TIME"]

# === Sets ===
V = list(range(25))  # 25 vehicles
C = list(range(1, len(data)))  # Customers (1 to n), excluding depot (0)
N = list(range(len(data)))  # All nodes including depot

# === Parameters ===
dc = data['DEMAND'].tolist()
dc[0] = 0  # Depot demand = 0
dc_pre = data_pre['DEMAND'].tolist()
dc_pre[0] = 0  # Depot demand = 0
coords = list(zip(data['XCOORD'], data['YCOORD']))
ai = data['READY_TIME'].tolist()
bi = data['DUE_DATE'].tolist()
si = data['SERVICE_TIME'].tolist()

# Problem parameters
cij = 30
delta = 0.17        #https://nrf.com/research/2024-consumer-returns-retail-industry
lambda_v = 7
pi = 10
theta = 0.7
mv = 100000
cv = 100
hv = 15
ev = 0.1
cf = 1.5
fv = 0.3
M = 1000000
w1, w2, w3 = 1, 1, 1  # weight for objectives

# === Distance matrix ===
t = np.zeros((len(N), len(N)))
for i in N:
    for j in N:
        t[i][j] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])

# === Build model function ===
def build_base_model():
    m = Model()
    x = m.addVars(N, N, V, vtype=GRB.BINARY, name="x")
    z = m.addVars(N, V, vtype=GRB.BINARY, name="z")
    y = m.addVars(V, vtype=GRB.BINARY, name="y")  
    L0 = m.addVars(V, vtype=GRB.CONTINUOUS, name="L0")
    L = m.addVars(C, V, vtype=GRB.CONTINUOUS, name="L")
    alpha = m.addVars(N, V, vtype=GRB.CONTINUOUS, name="alpha")
    beta = m.addVars(V, vtype=GRB.CONTINUOUS, name="beta")
    Fc = m.addVars(C, V, vtype=GRB.CONTINUOUS, name="Fc")
    Rc = m.addVars(C, V, vtype=GRB.CONTINUOUS, name="Rc")
    Qd = m.addVars(C, vtype=GRB.CONTINUOUS, name="Qd")
    Qr = m.addVars(C, vtype=GRB.CONTINUOUS, name="Qr")

    # === Constraints ===

    # Each vehicle begins its route in the depot and finishes in the depot
    for v in V:
        m.addConstr(quicksum(x[0, i, v] for i in N) == quicksum(x[i, 0, v] for i in N), 
                name='Fromtodepot_%s' % v)
    
    # Vehicle only depart from the depot if it is used
    for v in V:
        m.addConstr(quicksum(x[0, i, v] for i in C) == y[v], name='depart_%s' % v)
    
    # Vehicle can be used or not
    for v in V:
        m.addConstr(y[v] <= 1, name='usedORnot_%s' % v)

    # Flow conservation, vehicle enter node and must exit
    for v in V:
        for i in C:
            m.addConstr(quicksum(x[i, j, v] for j in C if j != i) == quicksum(x[j, i, v] for j in C if j != i), name='FlowConservation_%s_%s' % (i, v))
    
    # No self-loop on customer nodes 
    for v in V:
        for i in C:
            m.addConstr(x[i, i, v] == 0, 
                    name='NoSelfLoop_%s_%s' % (i, v)) 

    # Each destination node is visited from nodes that previously visited
    for v in V:
        for i in N:
            m.addConstr(quicksum(x[i, j, v] for j in N) == z[i, v], 
                    name='Visit_%s_%s' % (i, v))     
    
    # Each customer node can be visited by different vehicles
    for i in C: 
        m.addConstr(quicksum(z[i, v] for v in V) >= 1, 
                name='MultiVisit_%s' % i)
    
    # Customer forward demand may not be fully satisfied in each period
    for i in C:
        m.addConstr(quicksum(Fc[i, v] for v in V) + Qd[i] >= dc[i], 
                name='forward_%s' % i)
    
    # Customer forward demand is not served if vehicle does not visit 
    for i in C:
        for v in V:
            m.addConstr(Fc[i,v] <= dc[i] * z[i,v], name=f"FlowIfVisit_{i}_{v}")

    # Customer reverse demand which depends on the forward demand of the previous period may not be fully satisfied in current period 
    for i in C:
        m.addConstr(quicksum(Rc[i, v] for v in V) + Qr[i] >= dc_pre[i] * delta, 
                name='return_%s' % i)
    
    # Customer reverse demand is not served if vehicle does not visit 
    for i in C:
        for v in V:
            m.addConstr(Rc[i,v] <= dc_pre[i] * delta * z[i,v], name=f"ReturnIfVisit_{i}_{v}")

    # Each initial vehicle load is the accumulated demand of all customer points assigned to this vehicle
    for v in V:
        m.addConstr(L0[v] >= quicksum(Fc[i, v] for i in C), 
                name='startload_%s' % v)
    
    # Ensure the load balance of vehicles after they visit each customer point on their route
    for v in V:
        for i in C:
            for j in C:
                if i != j:
                    m.addConstr(L[j, v] >= L[i, v] - Fc[j, v] + Rc[j, v] - M * (1 - x[i, j, v]), name='otherload_%s_%s_%s' % (i, j, v))
    
    # The initial vehicle load doesn't exceed capacity of vehicle
    for v in V:
        m.addConstr(L0[v] <= lambda_v, 
                name='startcap_%s' % v)
    
    # The vehicle load after serving each customer doesn't exceed capacity of vehicle
    for v in V:
        for i in C:
            m.addConstr(L[i, v] <= lambda_v * z[i, v], 
                name='othercap_%s_%s' % (i, v))

    # Start of service time of destination node is after origin node and service duration    
    for v in V:
        for i in N:
            for j in N:
                if i != j:
                    m.addConstr(alpha[j, v] >= alpha[i, v] + si[i] + t[i][j] - M * (1 - x[i, j, v]), name='starttime_%s_%s_%s' % (i, j, v))
    
    # Time window constraints start of service time inside time window
    for v in V:
        for i in N:
            m.addConstr(alpha[i, v] >= ai[i] * z[i, v], 
                    name='EarliestStart_%s_%s' % (i, v)) 
            m.addConstr(alpha[i, v] <= bi[i] + M * (1 - z[i, v]), 
                    name='LatestStart_%s_%s' % (i, v)) 
    
    # Total duration that each vehicle finishes all services
    for v in V:
        m.addConstr(beta[v] == quicksum(t[i][j] * x[i, j, v] for i in N for j in C) + quicksum(si[i] * x[i, j, v] for i in C for j in C), 
                name='totalhour_%s' % v)
    
    # The maximum allowed transportation duration limit for each vehicle
    for v in V  :
        m.addConstr(beta[v] <= mv, 
                name='maxhour_%s' % v)
    
    return m, x, y, z, alpha, beta, Qd, Qr,Fc,Rc


# === Solve single-objective functions ===
def solve_feco_best():
    m, x, y, z, alpha, beta, Qd, Qr,Fc,Rc = build_base_model()
    Feco = quicksum(hv * beta[v] + cv * y[v] for v in V) + \
           quicksum(t[i][j] * x[i, j, v] * (cf * fv + cij) for i in N for j in N for v in V if i != j) + \
           pi * quicksum(Qd[i] + (1 - theta) * Qr[i] for i in C)
    m.setObjective(Feco, GRB.MINIMIZE)
    m.optimize()
    return m.objVal


def solve_fenv_best():
    m, x, y, z, alpha, beta, Qd, Qr,Fc,Rc = build_base_model()
    Fenv = quicksum(ev * t[i][j] * x[i, j, v] for i in N for j in N for v in V if i != j)
    m.setObjective(Fenv, GRB.MINIMIZE)
    m.optimize()
    return m.objVal

def solve_fsoc_best():
    m, x, y, z, alpha, beta, Qd, Qr,Fc,Rc = build_base_model()
    Fsoc = quicksum(beta[v] for v in V)
    m.setObjective(Fsoc, GRB.MINIMIZE)
    m.optimize()
    return m.objVal


# === Retrieve best values ===
Feco_best = solve_feco_best()
Fenv_best = solve_fenv_best()
Fsoc_best = solve_fsoc_best()

# === Final full model ===
m, x, y, z, alpha, beta, Qd, Qr,Fc,Rc = build_base_model()

Feco_best = 8126.91         # input best values as parameters !!!
Fenv_best = 20.7933         # input best values as parameters !!!
Fsoc_best = 99.6906         # input best values as parameters !!!

Feco = quicksum(hv * beta[v] + cv * y[v] for v in V) + \
       quicksum(t[i][j] * x[i, j, v] * (cf * fv + cij) for i in N for j in N for v in V if i != j) + \
       pi * quicksum(Qd[i] + (1 - theta) * Qr[i] for i in C)

Fenv = quicksum(ev * t[i][j] * x[i, j, v] for i in N for j in N for v in V if i != j)
Fsoc = quicksum(beta[v] for v in V)
Fsat = quicksum(theta * Qd[i] + (1 - theta) * Qr[i] for i in C)

F1 = w1 * (Feco - Feco_best) / Feco_best + \
     w2 * (Fenv - Fenv_best) / Fenv_best + \
     w3 * (Fsoc - Fsoc_best) / Fsoc_best

m.setObjectiveN(F1, index=0, priority=1, name="Sustainability")
m.setObjectiveN(Fsat, index=1, priority=0, name="CustomerSatisfaction")

# === Solve ===
m.optimize()

print("Final Objective Values:")
print("Feco:", Feco.getValue())
print("Fenv:", Fenv.getValue())
print("Fsoc:", Fsoc.getValue())
print("Customer Satisfaction (Fsat):", Fsat.getValue())

# === Output ===
if m.status == GRB.OPTIMAL:
    print("Optimal solution found.")
else:
    print("No optimal solution found.")

#========================================RESULTS========================================
m.update()
m.write('model.lp')
m.optimize()
m.write('model.sol')

arc_solution = m.getAttr('x', x)

# Extract routes from the solution
routes = {v: [] for v in V}
for v in V:
    for i in N:
        for j in N:
            if x[i, j, v].X > 0.5:
                routes[v].append((i, j))


xc = [c[0] for c in coords]
yc = [c[1] for c in coords]


# Print the objective value
print("Sustainability score:", m.getObjective(index=0).getValue())
print("Customer satisfaction:", m.getObjective(index=1).getValue())

# Print the matrix and x[i,j,v] and z[i,v]
for v in V:
    for i in N:
        print("\n")
        for j in N:
            print(f"x[{i},{j},{v}] = {arc_solution[i, j, v]:.2f}")
print("\n")

for v in V:
    for i in N:
        print(f"z[{i},{v}] = {z[i, v].X}")
print("\n")

# print matrix alpha[i,v]
for v in V:
    for i in N:
        print(f"alpha[{i},{v}] = {alpha[i, v].X}")
print("\n")

for v in V:
    print(f"Vehicle {v} service times:")
    sorted_routes = sorted(routes[v], key=lambda x: alpha[x[0], v].X)
    for (i, j) in sorted_routes:
        print(f"Node {i}: Service start at {alpha[i, v].X}")
    print("\n")

print("\n=== Forward Flow Fc[i,v] ===")
for i in C:
    for v in V:
        print(f"Fc[{i},{v}] = {m.getVarByName(f'Fc[{i},{v}]').X:.2f}", end="\t")
    print()

print("\n=== Return Flow Rc[i,v] ===")
for i in C:
    for v in V:
        print(f"Rc[{i},{v}] = {m.getVarByName(f'Rc[{i},{v}]').X:.2f}", end="\t")
    print()


print("\n=== Unmet Demand Qd[i] ===")
for i in C:
    print(f"Qd[{i}] = {Qd[i].X:.2f}")

print("\n=== Unmet Return Qr[i] ===")
for i in C:
    print(f"Qr[{i}] = {Qr[i].X:.2f}")


print("\n=== Load at Depot L0[v] ===")
for v in V:
    print(f"L0[{v}] = {m.getVarByName(f'L0[{v}]').X:.2f}")

print("\n=== Load at Customers L[i,v] ===")
for i in C:
    for v in V:
        print(f"L[{i},{v}] = {m.getVarByName(f'L[{i},{v}]').X:.2f}", end="\t")
    print()

print("\n=== Arrival Time alpha[i,v] ===")
for i in N:
    for v in V:
        print(f"alpha[{i},{v}] = {alpha[i,v].X:.2f}", end="\t")
    print()

print("\n=== Total Route Time beta[v] ===")
for v in V:
    print(f"beta[{v}] = {beta[v].X:.2f}")


#========================================PLOT========================================
# Plot the routes for only the first 3 vehicles
plt.figure(figsize=(15, 15))
plt.scatter(xc, yc, c='red')
plt.scatter(xc[0], yc[0], c='green', marker='s')  # depot

for i, (x, y) in enumerate(zip(xc, yc)):
    plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

colors = ['green', 'orange', 'blue']  # Only 3 colors needed
for v in range(3):  # Limit to first 3 vehicles
    for (i, j) in routes[v]:
        plt.arrow(xc[i], yc[i], xc[j] - xc[i], yc[j] - yc[i], 
                  color=colors[v], head_width=0.5, length_includes_head=True)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Route for Vehicle {v}')
    plt.show()



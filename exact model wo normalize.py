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
data = pd.read_fwf("RC104_25.txt", skiprows=8, header=None)
data.columns = ["CUST_NO", "XCOORD", "YCOORD", "DEMAND", "READY_TIME", "DUE_DATE", "SERVICE_TIME"]


# Load data from previous period
data_pre = pd.read_fwf("RC103_25.txt", skiprows=8, header=None)
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
delta = 0.17        #https://progresschamber.org/wp-content/uploads/2024/06/Chamber-of-Progress-Efficiency-and-Emissions-Impact-of-Last-Mile-Online-Delivery-in-the-US.pdf
lambda_v = 200
pi = 10
theta = 0.7
mv = 100000
cv = 100
hv = 15
ev = 0.1
cf = 0.5                                                                                        
fv = 0.1
M = 1000000
w1, w2, w3 = 1, 1, 1  # weight for objectives

dc_rev = [round(d * delta) for d in dc_pre]  # Integer reverse demand!!!!

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
    
    # Vehicle only depart from the depot if it is used
    for v in V:
        m.addConstr(quicksum(x[0, j, v] for j in N) == y[v], name='depart_%s' % v)
        
    # Vehicle only depart from the depot if it is used
    for v in V:
        m.addConstr(quicksum(x[i, 0, v] for i in N) == y[v], name='finish_%s' % v)
    
    # Vehicle can be used or not
    for v in V:
        m.addConstr(y[v] <= 1, name='usedORnot_%s' % v)

    # Flow conservation, vehicle enter node and must exit
    for v in V:
        for i in N:
            m.addConstr(quicksum(x[i, j, v] for j in N) == quicksum(x[j, i, v] for j in N), name='FlowConservation_%s_%s' % (i, v))
    
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
        m.addConstr(quicksum(Rc[i, v] for v in V) + Qr[i] >= dc_rev[i], 
                name='return_%s' % i)
    
    # Customer reverse demand is not served if vehicle does not visit 
    for i in C:
        for v in V:
            m.addConstr(Rc[i,v] <= dc_rev[i] * z[i,v], name=f"ReturnIfVisit_{i}_{v}")

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
            for j in C:
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

# === Final full model ===
m, x, y, z, alpha, beta, Qd, Qr,Fc,Rc = build_base_model()

Feco = quicksum(hv * beta[v] + cv * y[v] for v in V) + \
       quicksum(t[i][j] * x[i, j, v] * (cf * fv + cij) for i in N for j in N for v in V if i != j) + \
       pi * quicksum(Qd[i] + (1 - theta) * Qr[i] for i in C)

Fenv = quicksum(ev * t[i][j] * x[i, j, v] for i in N for j in N for v in V if i != j)
Fsoc = quicksum(beta[v] for v in V)
Fsat = quicksum(theta * Qd[i] + (1 - theta) * Qr[i] for i in C)

F1 = w1 * Feco + w2 * Fenv + w3 * Fsoc

m.setObjectiveN(F1, index=0, priority=1, name="Sustainability")
m.setObjectiveN(Fsat, index=1, priority=0, name="CustomerSatisfaction")

# === Solve ===
m.update()
m.write('VRPmodel.lp')
m.Params.timeLimit = 300
m.optimize()
m.write('VRPmodel.sol')


print("Final Objective Values:")
print("Feco:", Feco.getValue())
print("Fenv:", Fenv.getValue())
print("Fsoc:", Fsoc.getValue())
print("Customer Satisfaction (Fsat):", Fsat.getValue())

# Replace this with the actual metaheuristic result you obtained
Z_meta = 19071.62

# Make sure Gurobi found a solution before comparing
Z_exact = F1.getValue()

# Calculate Optimality Gap
opt_gap = ((Z_meta - Z_exact) / Z_exact) * 100

print(f"\nMetaheuristic Objective Value: {Z_meta:.2f}")
print(f"Gurobi Exact Objective Value: {Z_exact:.2f}")
print(f"Optimality Gap: {opt_gap:.2f}%")



# === Output ===
if m.status == GRB.OPTIMAL:
    print("Optimal solution found.")
else:
    print("No optimal solution found.")

# === Routing matrix x[i,j,v], only for used vehicles ===
#arc_solution = m.getAttr('x', x)
#for v in V:
#    if y[v].X > 0.5:  # Only print for used vehicles
#        print(f"\nRouting matrix x[i,j,v] for Vehicle {v}:")
#        header = "     " + "   ".join([f"{j:>5}" for j in N])  
#        print(header)
#        print("-" * len(header))  
#        for i in N:
#            row = [f"{arc_solution[i, j, v]:>5.2f}" for j in N]  
#            print(f"{i:>5}  " + "   ".join(row))  

# === z[i, v] matrix ===
for v in V:
    if y[v].X > 0.5:
        print(f"\nz[i, v] for Vehicle {v}:")
        for i in N:
            if z[i, v].X > 1e-4:
                print(f"  Node {i}: {z[i, v].X:.2f}")

# === Forward Demand Fc[i, v] ===
print("\n=== Forward Demand Fc[i, v] (Only Used Vehicles) ===")
for v in V:
    if y[v].X > 0.5:
        print(f"Vehicle {v} Forward Demand (Fc):")
        for i in C:
            val = Fc[i, v].X
            if val > 1e-4:
                print(f"  Customer {i}: {val:.2f}")
        print()

# === Reverse Demand Rc[i, v] ===
print("\n=== Reverse Demand Rc[i, v] (Only Used Vehicles) ===")
for v in V:
    if y[v].X > 0.5:
        print(f"Vehicle {v} Reverse Demand (Rc):")
        for i in C:
            val = Rc[i, v].X
            if val > 1e-4:
                print(f"  Customer {i}: {val:.2f}")
        print()

# === Service times and cumulative demand ===
routes = {v: [] for v in V}
for v in V:
    if y[v].X > 0.5:
        for i in N:
            for j in N:
                if x[i, j, v].X > 0.5:
                    routes[v].append((i, j))

        print(f"Vehicle {v} Service Times and Cumulative Demand:\n")
        print(f"{'Node':<8}{'Service Start Time':<20}{'Cumulative Demand':<20}")
        print("-" * 50)
        
        cumulative_demand = 0
        visited = set()
        sorted_routes = sorted(routes[v], key=lambda x: alpha[x[0], v].X)
        
        for (i, j) in sorted_routes:
            if i != 0 and i not in visited:
                visited.add(i)
                cumulative_demand += Fc[i, v].X
                print(f"{i:<8}{alpha[i, v].X:<20.2f}{cumulative_demand:<20.2f}")
        print()


#========================================PLOT========================================
# Plot the routes
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Coordinates
xc = [c[0] for c in coords]
yc = [c[1] for c in coords]

plt.figure(figsize=(15, 15))
plt.scatter(xc, yc, c='red', label='Customers')
plt.scatter(xc[0], yc[0], c='green', marker='s', label='Depot')

for i, (x, y_) in enumerate(zip(xc, yc)):
    plt.annotate(str(i), (x, y_), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

# Color map for up to 25 vehicles
cmap = plt.get_cmap('tab20', len(V))
used_vehicle_count = 0

for v in V:
    if y[v].X > 0.5:
        used_vehicle_count += 1
        color = cmap(v % 20)  # Cycle through 20 colors
        for (i, j) in routes[v]:
            plt.arrow(xc[i], yc[i], xc[j] - xc[i], yc[j] - yc[i],
                      color=color, head_width=0.4, length_includes_head=True, alpha=0.7)

        # Add label only once per vehicle
        plt.plot([], [], color=color, label=f'Vehicle {v}')

plt.xlabel('X Coordinate', fontsize=14)
plt.ylabel('Y Coordinate', fontsize=14)
plt.title(f'Routes of {used_vehicle_count} Used Vehicles', fontsize=18)
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

import point as pt

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}"
})

# Amount of Particles
N = 5000

# Amount of timesteps
T = 300

# Beta Parameter
beta = 20
#Simulation Timestep
dt = 0.1

#Rotation z-axis angle
alpha = 3 * np.pi / 4
#Matrix Q^tK
QtK = np.array([
    [np.cos(alpha), -np.sin(alpha), 0],
    [np.sin(alpha), np.cos(alpha), 0],
    [0, 0, -2],
])

#Matrix V
V = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])


#in: list of points
#out: array of dim (N,3) containing the coordinates of the points
def get_coords(temp_points):
    temp_location_array = np.zeros((len(temp_points), 3))
    for i, p in enumerate(temp_points):
        temp_location_array[i] = p.point
    return temp_location_array


#calculation of Z_b_i for point i
#In: point for which to calculate Z_b_i, pointsarray of pointlocations
#Out: Value of Z_b_i for point i and parameter beta
def zbi(temp_point, temp_location_array, _beta, _QtK):
    return np.sum(np.exp(_beta * np.matmul(temp_point.point, np.matmul(_QtK, temp_location_array.T))))


#In: array of every pointlocation, location_array, _beta
#Out: array of dim N containing the value of zbi for all points
def zbi_array(temp_points, temp_location_array, _beta, _QtK):
    temp_zbi_array = np.zeros(len(temp_points))
    for i, p in enumerate(temp_points):
        temp_zbi_array[i] = zbi(p, temp_location_array, _beta, _QtK)
    return temp_zbi_array


# in: object of type point
# out: object of type point, point at new location
def update_all_points(current_locations, _beta, _QtK, _V, _dt):
    """
    Updates all N points simultaneously using matrix operations.
    current_locations: (N, 3) array
    """
    # 1. Calculate interaction scores (the matrix version of your Z_bi logic)
    # Resulting 'scores' is (N, N) where scores[i, j] is the dot product of point i and point j
    scores = _beta * (current_locations @ _QtK @ current_locations.T)

    # 2. Calculate the exponential factors: (N, N) matrix
    exp_matrix = np.exp(scores)

    # 3. Calculate Z (the denominator) for each point: (N,) vector
    # Summing across the rows of the exp_matrix
    Z = np.sum(exp_matrix, axis=1, keepdims=True)

    # 4. Calculate the velocity term: (V @ X.T @ E.T).T
    # (3, 3) @ (3, N) @ (N, N) -> (3, N). Transpose to (N, 3)
    # Note: current_locations.T @ exp_matrix.T is the sum of points weighted by exp factors
    update_term = (_V @ current_locations.T @ exp_matrix.T).T

    # 5. Apply the update rule
    new_locations = current_locations + _dt * (update_term / Z)

    # 6. Normalize all rows to project back onto the unit sphere
    norms = np.linalg.norm(new_locations, axis=1, keepdims=True)
    return new_locations / norms


# (Random) Initialization of Points in the unitsphere
pointlist = [pt.Point() for i in range(N)]
point_array = get_coords(pointlist)

trajectories = np.zeros((T, N, 3))

#calculate trajectories
for lx in tqdm(range(T)):
    trajectories[lx] = point_array
    # Fully vectorized update step (no inner loops!)
    point_array = update_all_points(point_array, beta, QtK, V, dt)


#############

print("creating videos now...")

u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)

xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(left=0.1) # Adjusted back slightly since the text is gone

ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.2)

scat = ax.scatter([], [], [], s=10, color="crimson")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])
ax.axis("on")


def update(frame):
    pos = trajectories[frame]
    scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    return (scat,)


ani = FuncAnimation(fig, update, frames=T, interval=40, blit=True)

ani.save("beta20_3pi_over_4.gif", fps=60, dpi=200)

print(trajectories)
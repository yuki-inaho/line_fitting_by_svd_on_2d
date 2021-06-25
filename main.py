import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# True model: n_x x + n_y y + d = 0
n_x = 2
n_y = -5
n_norm = np.sqrt(n_x ** 2 + n_y ** 2)
d = 3
line_coeff_true = np.array([n_x, n_y, d]) / n_norm

# Generate point
N_points = 100
sigma = 0.05

x_range = [-1.0, 1.0]
y_range = [-1.0, 1.0]

points_x = np.random.rand(N_points) * (x_range[1] - x_range[0]) + x_range[0]
points_y = -(n_x * points_x + d) / n_y + np.random.normal(0, sigma, N_points)  # y = -(n_1 x + d) / n_2
points = np.vstack([points_x, points_y])  # shape[] = {2, N_points}
points_extend = np.vstack([points, np.repeat(1, N_points)])  # shape[] = {3, N_points}

# Line fitting
A = points_extend  # A.shape = {N_points, 3}
U, sigma, Vt = np.linalg.svd(A)  # U.shape = {3, 3}, len(sigma) = 3, Vt.shape = {N_points, N_points}
sigma_diag = np.diag(sigma)
Sigma = np.zeros((3, N_points))
Sigma[:3, :3] = sigma_diag
np.testing.assert_almost_equal(A, U @ Sigma @ Vt)
argmin_abs_sigma = np.argmin(np.abs(sigma))

line_coeff_estimated = U[:, argmin_abs_sigma]
line_normal_coeff_norm = np.sqrt(line_coeff_estimated[0] ** 2 + line_coeff_estimated[1] ** 2)
line_coeff_estimated /= line_normal_coeff_norm

print(f"line_coeff (True), n_x: {line_coeff_true[0]}, n_y: {line_coeff_true[1]}, d: {line_coeff_true[2]}")
print(
    f"line_coeff (Estimated), n_x: {line_coeff_estimated[0]}, n_y: {line_coeff_estimated[1]}, d: {line_coeff_estimated[2]}"
)

# Draw results
def predict_y(point_x, line_coeff):
    return -(line_coeff[0] * point_x + line_coeff[2]) / line_coeff[1]
predict_y_true = partial(predict_y, line_coeff=line_coeff_true)
predict_y_esitimated = partial(predict_y, line_coeff=line_coeff_estimated)

points_y_model_true = [predict_y_true(px) for px in points_x]
points_y_model_estimated = [predict_y_esitimated(px) for px in points_x]

plt.scatter(points_x, points_y, color="blue")
plt.plot(points_x, points_y_model_true, color="green", label="true")
plt.plot(points_x, points_y_model_estimated, color="red", label="estimated")
plt.legend(bbox_to_anchor=(1, 0), loc="lower right")

# plt.show()
plt.savefig("result.png")

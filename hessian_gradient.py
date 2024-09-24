import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x, y) = x^2 + y^2
def f(x):
    return x[0]**2 + x[1]**2

# Define the gradient of f: [df/dx, df/dy]
def gradient_f(x):
    df_dx = 2 * x[0]
    df_dy = 2 * x[1]
    return np.array([df_dx, df_dy])

# Find the direction of the steepest descent
def steepest_descent_direction(x):
    grad = gradient_f(x)
    # Normalize the negative gradient for the direction of steepest descent
    direction = -grad / np.linalg.norm(grad)
    return direction

# Example: Find the direction of steepest descent at point (1, 1)
x = np.array([1.0, 1.0])
direction = steepest_descent_direction(x)

print(f"Gradient at point {x}: {gradient_f(x)}")
print(f"Steepest descent direction at point {x}: {direction}")

# Visualize the function and the gradient direction
X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
Z = X**2 + Y**2

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, levels=20)
ax.quiver(x[0], x[1], -gradient_f(x)[0], -gradient_f(x)[1], color='r', scale=5, label='Steepest descent')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Contour plot with steepest descent direction')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Fake Data (e.g., Stock Price vs. Time)
# Let's say the true relationship is y = 2x + 1
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) # y = 3x + 4 + noise

# 2. Initialize Parameters (Random Guess)
# We want to find 'm' (slope) and 'b' (intercept)
m = np.random.randn(1)
b = np.random.randn(1)

learning_rate = 0.1
iterations = 1000

print(f"Starting Guess: m={m[0]:.2f}, b={b[0]:.2f}")

# 3. The Gradient Descent Loop
for iteration in range(iterations):
    # A. Make Prediction
    y_pred = m * X + b
    
    # B. Calculate Error (Mean Squared Error)
    # This is just for us to see, the machine needs the Gradient
    error = np.mean((y_pred - y) ** 2)
    
    # C. Calculate Gradients (The Math Part)
    # --------------------------------------
    # QUESTION: What are dm and db?
    # ANSWER: They are the "Slope of the Error Mountain" with respect to m and b.
    # If m_gradient is positive, it means increasing m will INCREASE error.
    # So we want to go the OPPOSITE direction (subtract the gradient).
    
    # DERIVATION (Chain Rule):
    # 1. Loss Function (MSE):  E = (1/N) * sum( (y_pred - y)^2 )
    # 2. Prediction:           y_pred = m*x + b
    # 
    # We want to find dE/dm (how much Error changes if we wiggle m).
    # dE/dm = dE/dy_pred * dy_pred/dm
    # 
    # Step A: dE/dy_pred = 2 * (y_pred - y)      [Power Rule: d(u^2)/du = 2u]
    # Step B: dy_pred/dm = x                     [Derivative of (mx+b) with respect to m is x]
    # 
    # Combine them:
    # dE/dm = (1/N) * sum( 2 * (y_pred - y) * x )
    #       = (2/N) * sum( x * (y_pred - y) )
    
    N = len(X)
    m_gradient = (2/N) * np.sum(X * (y_pred - y))  # This is dE/dm
    b_gradient = (2/N) * np.sum(y_pred - y)        # This is dE/db (since dy_pred/db = 1)
    
    # D. Update Parameters (The "Step Down the Mountain")
    m = m - (learning_rate * m_gradient)
    b = b - (learning_rate * b_gradient)
    
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: Error={error:.4f}, m={m[0]:.2f}, b={b[0]:.2f}")

print(f"\nFinal Result: m={m[0]:.2f}, b={b[0]:.2f}")
print("True Values:  m=3.00, b=4.00")

# 4. Visualization (Optional)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, m*X + b, color='red', label='Our Model')
plt.legend()
plt.title("Gradient Descent from Scratch")
plt.show()

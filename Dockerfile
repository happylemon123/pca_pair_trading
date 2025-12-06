# 1. The Base Image: Start with a computer that already has Python installed
FROM python:3.9-slim

# 2. The Setup: Create a folder inside the container
WORKDIR /app

# 3. The Dependencies: Install NumPy and Matplotlib
# (In a real project, you'd copy requirements.txt first)
RUN pip install numpy matplotlib

# 4. The Code: Copy your script from your laptop to the container
COPY gradient_descent_scratch.py .

# 5. The Command: What to do when the container starts?
CMD ["python", "gradient_descent_scratch.py"]

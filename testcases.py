import numpy as np

# Change this to 599 or 510
size = 599 

# Create a baseline of 10W
data = np.full(size, 10.0)

# Add a "Fridge Spike" (200W) in the middle 200 samples
data[200:400] = 200.0 

# Add some random noise
data += np.random.normal(0, 2, size)

# Print as a comma-separated string for Streamlit
print(",".join(data.astype(str)))
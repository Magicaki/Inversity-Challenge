import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Load the Image
number = input('Which image do you want to use? (1, 2, 3): ')
image_path = f"inversity{number}.png"  # Ensure correct filename
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Error: Image not loaded. Check the file path!")

print(f'Generating path for image {number}...')

# Convert to HSV for Better Color Detection
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Detect Red Paths (Allowed Movement Areas)
lower_red1 = np.array([0, 80, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 80, 50])
upper_red2 = np.array([180, 255, 255])

mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# Fill Gaps in the Red Path (Larger Dilation)
kernel = np.ones((5, 5), np.uint8)
mask_red = cv2.dilate(mask_red, kernel, iterations=1)

# Detect Orange Blobs (Obstacles)
lower_orange = np.array([5, 50, 50])
upper_orange = np.array([20, 255, 255])
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

# Create a Walkable Grid
grid = np.zeros(image.shape[:2], dtype=np.uint8)
grid[mask_red > 0] = 1
grid[mask_orange > 0] = 0  # Block orange blobs

# Add Buffer Around Orange Blobs
buffer_size = 5
for y, x in zip(*np.where(mask_orange > 0)):
    grid[max(0, y-buffer_size):min(y+buffer_size, grid.shape[0]), 
         max(0, x-buffer_size):min(x+buffer_size, grid.shape[1])] = 0

# Find All Red Path Points
red_path_points = np.column_stack(np.where(mask_red > 0))

# Determine Start (Top-Left) and End (Top-Right) Points
start_candidates = red_path_points[red_path_points[:, 1] == np.min(red_path_points[:, 1])]
start_point = tuple(start_candidates[np.argmin(start_candidates[:, 0])])  # Top-left corner

end_candidates = red_path_points[red_path_points[:, 1] == np.max(red_path_points[:, 1])]
end_point = tuple(end_candidates[np.argmin(end_candidates[:, 0])])  # Top-right corner

# Create Graph That Only Allows Movement on Red Paths
graph = nx.grid_2d_graph(*grid.shape)
for y, x in zip(*np.where(grid == 0)):
    if graph.has_node((y, x)):
        graph.remove_node((y, x))

# Check if start and end are walkable
if grid[start_point] == 0:
    print("Warning: Start point is not on a valid red path!")
if grid[end_point] == 0:
    print("Warning: End point is not on a valid red path!")

# Check connectivity
if not nx.has_path(graph, start_point, end_point):
    print("No path found. Check for gaps in the grid.")

# Find the Shortest Path
if nx.has_path(graph, start_point, end_point):
    path = nx.shortest_path(graph, start_point, end_point)
else:
    path = None

# Visualize the Path
path_image = image.copy()
if path:
    for y, x in path:
        # Draw a slightly larger circle for each path point to widen the path
        cv2.circle(path_image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
else:
    print("No path found that strictly follows the red lines.")

plt.figure(figsize=(10, 5))
plt.title("Strict Path (Only Following Red Lines, Avoiding Orange)")
plt.imshow(cv2.cvtColor(path_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

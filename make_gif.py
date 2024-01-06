from PIL import Image
import os

# Folder containing PNG files
folder_path = '/Users/hyejunlee/fcos_3d/output_waymo_sim_depth_hpc_40'

# Output GIF file name
output_gif_path = 'output_waymo_sim_depth_hpc_40.gif'

# Read all PNG files in folder
png_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith('.png')]

# Load images
images = [Image.open(png) for png in png_files]

# Save the images as a GIF
images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)

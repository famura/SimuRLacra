# This script is used to turn off the motion blur plugin from RenderPipeline
# We do this by commenting out the motion blur plugin in the plugins yaml configuration file from RP

import os.path as osp

fname = "plugins.yaml"
rp_dir = osp.join(osp.dirname(__file__), "thirdParty", "render_pipeline", "config")

# Read each line of the file into an array
with open(osp.join(rp_dir, fname), 'r') as file:
    lines = file.readlines()

i = 0
for line in lines:
    # If motion blur is already turned off
    if "# - motion_blur\n" in str(line):
        print("RenderPipeline motion blur already turned off")
        break
    # Turning motion blur off by commenting out
    if "- motion_blur" in str(line):
        lines[i] = "    # - motion_blur\n"
        print("Turning off RenderPipeline motion blur")
        break
    i += 1

# Overwrite lines of the file with lines from the array
with open(osp.join(rp_dir, fname), 'w') as file:
    file.writelines(lines)

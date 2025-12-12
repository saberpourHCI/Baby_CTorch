import imageio
import glob

frames = sorted(glob.glob("../c/code/png_frames/frame_*.png"))

images = [imageio.imread(f) for f in frames]

imageio.mimsave("training_progress.gif", images, duration=0.2)

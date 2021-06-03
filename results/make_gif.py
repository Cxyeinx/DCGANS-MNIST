import imageio
import os

images = []

for filename in sorted(os.listdir()):
    if filename.endswith(".png"):
        print(filename)
        images.append(imageio.imread(filename))

imageio.mimsave('movie.gif', images)
    
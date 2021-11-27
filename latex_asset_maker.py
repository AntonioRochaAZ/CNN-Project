import matplotlib
matplotlib.rcParams.update({
    "text.usetex": True
})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
from PIL import Image
from datasets import HASYv2Dataset
from warnings import warn

###############################################################
warn("This code does not work for all assets, some have to be"
     "treated differently. All assets are available in the"
     "_Assets folder", RuntimeWarning)
###############################################################


fig = plt.figure(figsize=(6, 6))

# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(2.5), Size.Fixed(0.1)]
v = [Size.Fixed(0.1), Size.Fixed(0.1)]

divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
# The width and height of the rectangle are ignored.

ax = fig.add_axes(divider.get_position(),
                  axes_locator=divider.new_locator(nx=1, ny=1))

plt.plot(1, 1)
fig = plt.gcf()
fig.patch.set_facecolor('#747dcf')

ltx_dict = HASYv2Dataset.latex_dict
for index in ltx_dict:
    name = ltx_dict[index]
    try:
        plt.title(r'${'+f"{name}"+'}$', fontsize=100, pad=200)
        if name.islower():
            filename = "_" + name
        else:
            filename = name
        plt.savefig(f".//_Assets/{filename}.png")
        img = Image.open(f".//_Assets/{filename}.png")
        img.crop((360, 300, 855, 745)).save(f".//_Assets/{filename}.png")
    except:
        print(f"{name} didn't work, skipped.")
        continue

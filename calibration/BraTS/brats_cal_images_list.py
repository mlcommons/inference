import numpy as np
np.random.seed(0)
images = []
for i in [0, 2, 3, 4]:
    with open("../../v0.7/medical_imaging/3d-unet/folds/fold{:d}_validation.txt".format(i)) as f:
        for line in f:
            images.append(line.rstrip())
indices = np.random.permutation(len(images))[:40]
selected = sorted([images[idx] for idx in indices])
with open("brats_cal_images_list.txt", "w") as f:
    for img in selected:
        print(img, file=f)
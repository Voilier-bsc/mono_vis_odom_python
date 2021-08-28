import numpy as np

base_img = np.zeros((3,1024,2048),dtype=np.uint8)
print(base_img.shape)
print(base_img[:,324:700,404:1645].shape)


# [324:701][404:1646]
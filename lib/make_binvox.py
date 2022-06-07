import os
from subprocess import call

path = '/home/ywlee/Dataset/Historical_Sites/train'
dnames = os.listdir(path)

for i, dn in enumerate(dnames):
    binvox = os.path.join(os.path.join(path, dn), "{}.binvox".format(dn))
    if os.path.exists(binvox):
        os.remove(binvox)

    fpath = os.path.join(os.path.join(path, dn), "{}.obj".format(dn))
    call(['binvox', fpath])
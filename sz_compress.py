import os
for i in range(1520,1550):
    filename="aramco-snapshot-%s.f32" % str(i)
    command="/home/jliu447/packs/SZ/bin/sz -z -f -M REL -R 1e-2 -i aramco/%s -3 235 449 449" % filename
    os.system(command) 
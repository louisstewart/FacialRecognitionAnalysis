#!/usr/bin/env python

import os
import shutil
import sys
import random

args = sys.argv

if len(args) < 2:
    print("usage: %s <input_dir>" % args[0])
    sys.exit(1)

in_dir = args[1]
current = os.getcwd()
current = os.path.join(current, in_dir)

if not os.path.exists(current):
    raise IOError("Image folder not found")

os.chdir(current)

print in_dir
num = in_dir.split(" ")[1]
train = num+"_train"
test = num+"_test"
cv = num+"_cv"
trash = "bin"


def index_ne(xs, ind):
    try:
        return xs.index(ind)
    except:
        return -1


index = 0

angles = []
for pitch in range(-15, 20, 5):
    angles.append((pitch, 0))

for yaw in range(-15, 20, 5):
    angles.append((0, yaw))

# add corners
angles.append((-10, -10))
angles.append((10, -10))
angles.append((-10, 10))
angles.append((10, 10))

length = len(angles)

# Make temp directories to hold all images for each yaw-pitch pair
direc = []
for i in range(1,length+1):
    if not os.path.exists(os.path.join(current, repr(i))):
        os.mkdir(os.path.join(current, repr(i)))
    direc.append(repr(i))

if not os.path.exists(os.path.join(current, train)):
    os.mkdir(os.path.join(current, train))  # Create manifold directory.
if not os.path.exists(os.path.join(current, test)):
    os.mkdir(os.path.join(current, test))  # Create manifold directory.
if not os.path.exists(os.path.join(current, cv)):
    os.mkdir(os.path.join(current, cv))  # Create manifold directory.
if not os.path.exists(os.path.join(current, trash)):
    os.mkdir(os.path.join(current, trash))


count = 0
to_move = 0
p_id = ""

drop_count = 0
drop_max = 100


for subdir, dirs, files in os.walk('./'):
    for f in files:
        orig_path = os.path.join(subdir, f)
        s = f.split("_")
        if len(s) < 6:
            continue
        if s[0] == ".":
            del s[0]

        # File name = id_dist_pitch_yaw_roll_lights
        rot_str = (int(s[2]), int(s[3]))

        i = index_ne(angles, rot_str)  # Check this is a tuple we are looking for.
        if i > -1 and index_ne(direc, repr(i)) > -1:
            # i will be the name of the new folder we put the image in
            path = "_".join(s)
            shutil.move(orig_path, os.path.join(current, direc[i], path))  # Move the file.

        count += 1


# Walk the temp directories and move the files out into the train/test/cv folders.
dir_set = set(direc)
four = random.sample(dir_set, 4)
trs = list(dir_set - set(four))
tes = four[:2]
cvs = four[2:]

for dir_n in direc:
    os.chdir(os.path.join(current, dir_n))
    for subdir, dirs, files in os.walk("./"):
        for fin in files:
            orig_path = os.path.join(subdir, fin)
            if dir_n in trs:
                shutil.move(orig_path, os.path.join(current, train, fin))
            elif dir_n in tes:
                shutil.move(orig_path, os.path.join(current, test, fin))
            elif dir_n in cvs:
                shutil.move(orig_path, os.path.join(current, cv, fin))

os.chdir(current)
for dire in direc:
    os.rmdir(os.path.join(current, dire))

#!/usr/bin/env python

import os
import shutil
import random
import glob
import sys

args = sys.argv

drop_count = 0
drop_max = 100

if len(args) > 1:
    drop_max = int(args[1])

current = os.getcwd()

train_img = glob.glob("*_train")
test_img = glob.glob("*_test")
cv_img = glob.glob("*_cv")


train = "train"
test = "test"
cv = "cv"
trash = "bin"
if not os.path.exists(os.path.join(current, train)):
    os.mkdir(os.path.join(current, train))
if not os.path.exists(os.path.join(current, test)):
    os.mkdir(os.path.join(current, test))
if not os.path.exists(os.path.join(current, cv)):
    os.mkdir(os.path.join(current, cv))
if not os.path.exists(os.path.join(current, trash)):
    os.mkdir(os.path.join(current, trash))


for image in train_img:
    os.chdir(os.path.join(current, image))
    for subdir, dirs, files in os.walk('./'):
        for f in files:
            orig_path = os.path.join(subdir, f)
            if drop_count < drop_max and random.random() < 0.009 and os.path.exists(orig_path):
                tok = f.split("_")
                search = "_".join(tok[:6])
                others = glob.glob(search + "*")
                if len(others) == 3:
                    for s in others:
                        try:
                            shutil.move(os.path.join(subdir, s), os.path.join(current, trash, s))
                        except IOError:
                            print search
                    drop_count += 1
                else:
                    shutil.move(orig_path, os.path.join(current, train, f))
            else:
                if os.path.exists(orig_path):
                    shutil.move(orig_path, os.path.join(current, train, f))

for image in test_img:
    os.chdir(os.path.join(current, image))
    for subdir, dirs, files in os.walk('./'):
        for f in files:
            orig_path = os.path.join(subdir, f)
            shutil.move(orig_path, os.path.join(current, test, f))

for image in cv_img:
    os.chdir(os.path.join(current, image))
    for subdir, dirs, files in os.walk('./'):
        for f in files:
            orig_path = os.path.join(subdir, f)
            shutil.move(orig_path, os.path.join(current, cv, f))

os.chdir(current)
for image in train_img:
    os.rmdir(image)
for image in test_img:
    os.rmdir(image)
for image in cv_img:
    os.rmdir(image)

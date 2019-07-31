    Suppose we have a voc-formated datasets. but for some reason, those origin datasets' image resolution is huge, especially compared to its gtboxes. Those huge gap between image resolution and its gtbox resolution will make the model sick. can't get the positive loss down. only get a serial of backgroud negtive boxes' loss down.which will get the training failed.
    To avoid thoes bad situations, we would like to crop some small patches around the gtboxes from the original images.aka patches 300x300. then resize to 128x128.
    1. using `3_hd_gen_patches_voc.py` to do the patching.

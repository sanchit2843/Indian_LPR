import os
import glob
import shutil

for idx, i in enumerate(
    glob.glob(
        "/media/sanchit/current_working_datasets/idd/idd-segmentation/IDD_Segmentation/leftImg8bit/*/*/*.png"
    )
):
    shutil.copyfile(
        i,
        os.path.join(
            "/media/sanchit/current_working_datasets/idd/images", str(idx) + ".png"
        ),
    )

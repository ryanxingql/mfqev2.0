"""Extract training set.

Randomly select frame to patch.
Patches are stored in several npys.
Each npy contains several batches.
So there are n x batch_size patches in each npy.
Return: a few npy with shape (n x width_patch x width_height x 1), dtype=np.float32 \in [0,1]."""
import os, glob, gc, h5py
import numpy as np
import random, math


def y_import(video_path, height_frame, width_frame, nfs, startfrm, bar=True, opt_clear=True):
    """Import Y channel from a yuv video.

    startfrm: start from 0
    return: (nfs * height * width), dtype=uint8"""

    fp = open(video_path, 'rb')

    # target at startfrm
    blk_size = int(height_frame * width_frame * 3 / 2)
    fp.seek(blk_size * startfrm, 0)

    d0 = height_frame // 2
    d1 = width_frame // 2

    Yt = np.zeros((height_frame, width_frame), dtype=np.uint8) # 0-255

    for ite_frame in range(nfs):

        for m in range(height_frame):
            for n in range(width_frame):
                Yt[m,n] = ord(fp.read(1))
        for m in range(d0):
            for n in range(d1):
                fp.read(1)
        for m in range(d0):
            for n in range(d1):
                fp.read(1)

        if ite_frame == 0:
            Y = Yt[np.newaxis, :, :]
        else:
            Y = np.vstack((Y, Yt[np.newaxis, :, :]))

        if bar:
            print("\r%4d | %4d" % (ite_frame + 1, nfs), end="", flush=True)
    if opt_clear:
        print("\r                                      ", end="\r")

    fp.close()
    return Y


def func_PatchFrame(info_patch, num_patch, ite_npy, mode):
    """Patch and store four npys with a same index.

    Shuffle the patches inside these four npys before saving."""
    
    order_FirstFrame, order_FirstPatch, order_LastFrame, order_LastPatch, list_CmpVideo, \
    VideoIndex_list_list, MidIndex_list_list, PreIndex_list_list, SubIndex_list_list, dir_save_stack = info_patch[:]

    ### Init stack
    stack_pre = np.zeros((num_patch, height_patch, width_patch, 1), dtype=np.float32)
    stack_cmp = np.zeros((num_patch, height_patch, width_patch, 1), dtype=np.float32)
    stack_sub = np.zeros((num_patch, height_patch, width_patch, 1), dtype=np.float32)
    stack_raw = np.zeros((num_patch, height_patch, width_patch, 1), dtype=np.float32)

    ### Extract patches
    cal_patch_total = 0

    num_frame_total = order_LastFrame - order_FirstFrame + 1

    for ite_frame, order_frame in enumerate(range(order_FirstFrame, order_LastFrame + 1)):

        print("\rframe %d | %d" % (ite_frame + 1, num_frame_total), end="")

        cal_patch_frame = 0

        ### Extract basic information
        index_video = VideoIndex_list_list[order_frame]
        index_Mid = MidIndex_list_list[order_frame]
        index_Pre = PreIndex_list_list[order_frame]
        index_Sub = SubIndex_list_list[order_frame]

        cmp_path = list_CmpVideo[index_video]
        cmp_name = cmp_path.split("/")[-1].split(".")[0]
        raw_name = cmp_name
        raw_name = raw_name + ".yuv"
        raw_path = os.path.join(dir_raw, raw_name)

        dims_str = raw_name.split("_")[1]
        width_frame = int(dims_str.split("x")[0])
        height_frame = int(dims_str.split("x")[1])

        ### Cal step
        step_height = int((height_frame - height_patch) / (num_patch_height - 1))
        step_width = int((width_frame - width_patch) / (num_patch_width - 1))

        ### Load frames
        Y_raw = np.squeeze(y_import(raw_path, height_frame, width_frame, 1, index_Mid, bar=False, opt_clear=False))
        Y_cmp = np.squeeze(y_import(cmp_path, height_frame, width_frame, 1, index_Mid, bar=False, opt_clear=False))
        Y_pre = np.squeeze(y_import(cmp_path, height_frame, width_frame, 1, index_Pre, bar=False, opt_clear=False))
        Y_sub = np.squeeze(y_import(cmp_path, height_frame, width_frame, 1, index_Sub, bar=False, opt_clear=False))

        ### Patch
        for ite_patch_height in range(num_patch_height):

            start_height = ite_patch_height * step_height

            for ite_patch_width in range(num_patch_width):

                if (order_frame == order_FirstFrame) and (cal_patch_frame < order_FirstPatch):
                    cal_patch_frame += 1
                    continue
                if (order_frame == order_LastFrame) and (cal_patch_frame > order_LastPatch):
                    cal_patch_frame += 1
                    continue

                start_width = ite_patch_width * step_width

                stack_pre[cal_patch_total, 0:height_patch, 0:width_patch, 0] = Y_pre[start_height:(start_height+height_patch), start_width:(start_width+width_patch)] / 255.0
                stack_cmp[cal_patch_total, 0:height_patch, 0:width_patch, 0] = Y_cmp[start_height:(start_height+height_patch), start_width:(start_width+width_patch)] / 255.0
                stack_sub[cal_patch_total, 0:height_patch, 0:width_patch, 0] = Y_sub[start_height:(start_height+height_patch), start_width:(start_width+width_patch)] / 255.0
                stack_raw[cal_patch_total, 0:height_patch, 0:width_patch, 0] = Y_raw[start_height:(start_height+height_patch), start_width:(start_width+width_patch)] / 255.0

                cal_patch_total += 1
                cal_patch_frame += 1

    ### Shuffle and save npy
    print("\nsaving 1/4...", end="")
    random.seed(100)
    random.shuffle(stack_pre)
    save_path = os.path.join(dir_save_stack, "stack_" + mode + "_pre_" + str(ite_npy) + ".hdf5")
    f = h5py.File(save_path, "w")
    f.create_dataset('stack_pre', data=stack_pre)
    f.close()
    stack_pre = []
    gc.collect()

    print("\rsaving 2/4...", end="")
    random.seed(100)
    random.shuffle(stack_cmp)
    save_path = os.path.join(dir_save_stack, "stack_" + mode + "_cmp_" + str(ite_npy) + ".hdf5")
    f = h5py.File(save_path, "w")
    f.create_dataset('stack_cmp', data=stack_cmp)
    f.close()
    stack_cmp = []
    gc.collect()

    print("\rsaving 3/4...", end="")
    random.seed(100)
    random.shuffle(stack_sub)
    save_path = os.path.join(dir_save_stack, "stack_" + mode + "_sub_" + str(ite_npy) + ".hdf5")
    f = h5py.File(save_path, "w")
    f.create_dataset('stack_sub', data=stack_sub)
    f.close()
    stack_sub = []
    gc.collect()

    print("\rsaving 4/4...", end="")
    random.seed(100)
    random.shuffle(stack_raw)
    save_path = os.path.join(dir_save_stack, "stack_" + mode + "_raw_" + str(ite_npy) + ".hdf5")
    f = h5py.File(save_path, "w")
    f.create_dataset('stack_raw', data=stack_raw)
    f.close()
    stack_raw = []
    gc.collect()

    print("\r                   ", end="\r") # clear bar


def main_extract_TrainingSet():
    """Extract training setself.

    Select a non-PQF between each pair of PQFs.
    Randomly select up to 20 non-PQFs each video."""

    for QP in QP_list:
    
        dir_cmp = dir_cmp_pre + str(QP)
        dir_PQFLabel = dir_PQFLabel_pre + str(QP)
        
        ### List all cmp video
        list_CmpVideo = glob.glob(os.path.join(dir_cmp, "*.yuv"))
        num_CmpVideo = len(list_CmpVideo)

        ### Init dir_save_stack for this QP
        dir_save_stack = dir_save_stack_pre + str(QP)
        if not os.path.exists(dir_save_stack):
            os.makedirs(dir_save_stack)

        ### List all randomly selected non-PQFs with their pre/sub PQFs and calculate the num of patches
        VideoIndex_list_list = []
        MidIndex_list_list = []
        PreIndex_list_list = []
        SubIndex_list_list = []
        
        cal_frame = 0

        for ite_CmpVideo in range(num_CmpVideo): # video by video

            cmp_name = list_CmpVideo[ite_CmpVideo].split("/")[-1].split(".")[0]

            # load PQF label
            PQFLabel_path = os.path.join(dir_PQFLabel, "PQFLabel_" + cmp_name + PQFLabel_sub)
            PQF_label = h5py.File(PQFLabel_path,'r')['PQF_label'][:]   

            # locate PQFs
            PQFIndex_list = [i for i in range(len(PQF_label)) if PQF_label[i] == 1]
            num_PQF = len(PQFIndex_list)
            
            #
            MidIndex_list = PQFIndex_list[1: (num_PQF - 1)]
            PreIndex_list = PQFIndex_list[0: (num_PQF - 2)]
            SubIndex_list = PQFIndex_list[2: num_PQF]
            
            # randomly select maximum allowable pairs
            random.seed(666)
            random.shuffle(PreIndex_list)
            random.seed(666)
            random.shuffle(SubIndex_list)
            random.seed(666)
            random.shuffle(MidIndex_list)
            
            num_pairs = len(PreIndex_list)
            if num_pairs > max_NonPQF_OneVideo:
                PreIndex_list = PreIndex_list[0: max_NonPQF_OneVideo]
                SubIndex_list = SubIndex_list[0: max_NonPQF_OneVideo]
                MidIndex_list = MidIndex_list[0: max_NonPQF_OneVideo]
                

            # record
            cal_frame += len(PreIndex_list)
            VideoIndex_list_list += [ite_CmpVideo] * len(PreIndex_list) # video index for all selected non-PQFs
            PreIndex_list_list += PreIndex_list
            MidIndex_list_list += MidIndex_list
            SubIndex_list_list += SubIndex_list

        num_patch_available = cal_frame * num_patch_PerFrame
        print("Available frames: %d - patches: %d" % (cal_frame, num_patch_available))

        ### Shuffle the numbering of all frames
        random.seed(888)
        random.shuffle(VideoIndex_list_list)
        random.seed(888)
        random.shuffle(MidIndex_list_list)
        random.seed(888)
        random.shuffle(PreIndex_list_list)
        random.seed(888)
        random.shuffle(SubIndex_list_list)

        ### Cut down the num of frames
        max_patch_total = int(num_patch_available / batch_size) * batch_size
        max_frame_total = math.ceil(max_patch_total / num_patch_PerFrame) # may need one more frame to patch

        VideoIndex_list_list = VideoIndex_list_list[0: max_frame_total]
        MidIndex_list_list = MidIndex_list_list[0: max_frame_total]
        PreIndex_list_list = PreIndex_list_list[0: max_frame_total]
        SubIndex_list_list = SubIndex_list_list[0: max_frame_total]

        ### Cal num of batch for each npy, including training and validation
        num_patch_val = int(int((1 - ratio_training) * max_patch_total) / batch_size) * batch_size
        num_patch_tra = max_patch_total - num_patch_val # we can make sure that it is a multiple of batch size

        num_batch_tra = int(num_patch_tra / batch_size)
        num_batch_val = int(num_patch_val / batch_size)

        num_npy_tra = int(num_batch_tra / max_batch_PerNpy)
        num_batch_PerNpy_list_tra = [max_batch_PerNpy] * num_npy_tra
        if (num_batch_tra % max_batch_PerNpy) > 0:
            num_batch_PerNpy_list_tra.append(num_batch_tra - max_batch_PerNpy * num_npy_tra)

        num_npy_val = int(num_batch_val / max_batch_PerNpy)
        num_batch_PerNpy_list_val = [max_batch_PerNpy] * num_npy_val
        if (num_batch_val % max_batch_PerNpy) > 0:
            num_batch_PerNpy_list_val.append(num_batch_val - max_batch_PerNpy * num_npy_val)

        ### Patch and stack
        # some frames may be partly patched.
        for ite_npy_tra in range(len(num_batch_PerNpy_list_tra)):

            print("stacking tra npy %d / %d..." % (ite_npy_tra + 1, len(num_batch_PerNpy_list_tra)))

            # Cal the position of the first patch and the last patch of this npy
            first_patch_cal = sum(num_batch_PerNpy_list_tra[0: ite_npy_tra]) * batch_size + 1
            order_FirstFrame = math.ceil(first_patch_cal / num_patch_PerFrame) - 1
            order_FirstPatch = first_patch_cal - order_FirstFrame * num_patch_PerFrame - 1

            last_patch_cal = sum(num_batch_PerNpy_list_tra[0: ite_npy_tra + 1]) * batch_size
            order_LastFrame = math.ceil(last_patch_cal / num_patch_PerFrame) - 1
            order_LastPatch = last_patch_cal - order_LastFrame * num_patch_PerFrame - 1

            # patch
            num_patch = num_batch_PerNpy_list_tra[ite_npy_tra] * batch_size
            info_patch = (order_FirstFrame, order_FirstPatch, order_LastFrame, order_LastPatch, list_CmpVideo, \
                          VideoIndex_list_list, MidIndex_list_list, PreIndex_list_list, SubIndex_list_list, dir_save_stack)
            func_PatchFrame(info_patch, num_patch=num_patch, ite_npy=ite_npy_tra, mode="tra")

        for ite_npy_val in range(len(num_batch_PerNpy_list_val)):

            print("stacking val npy %d / %d..." % (ite_npy_val + 1, len(num_batch_PerNpy_list_val)))

            # Cal the position of the first patch and the last patch of this npy
            first_patch_cal = (sum(num_batch_PerNpy_list_tra) + sum(num_batch_PerNpy_list_val[0: ite_npy_val])) * batch_size + 1
            order_FirstFrame = math.ceil(first_patch_cal / num_patch_PerFrame) - 1
            order_FirstPatch = first_patch_cal - order_FirstFrame * num_patch_PerFrame - 1

            last_patch_cal = (sum(num_batch_PerNpy_list_tra) + sum(num_batch_PerNpy_list_val[0: ite_npy_val + 1])) * batch_size
            order_LastFrame = math.ceil(last_patch_cal / num_patch_PerFrame) - 1
            order_LastPatch = last_patch_cal - order_LastFrame * num_patch_PerFrame - 1

            # patch
            num_patch = num_batch_PerNpy_list_val[ite_npy_val] * batch_size
            info_patch = (order_FirstFrame, order_FirstPatch, order_LastFrame, order_LastPatch, list_CmpVideo, \
                          VideoIndex_list_list, MidIndex_list_list, PreIndex_list_list, SubIndex_list_list, dir_save_stack)
            func_PatchFrame(info_patch, num_patch=num_patch, ite_npy=ite_npy_val, mode="val")


if __name__ == '__main__':

    QP_list = [32,42]

    ### Settings
    num_patch_width = 26
    num_patch_height = 16
    height_patch = 64
    width_patch = 64
    
    num_patch_PerFrame = num_patch_width * num_patch_height

    dir_database = "/home/x/SCI_1/Database/"
    dir_raw = os.path.join(dir_database, "train_108/raw")
    dir_cmp_pre = os.path.join(dir_database, "train_108/LDP_HM16.5/QP")
    
    dir_PQFLabel_pre = "/home/x/SCI_1/MFQEv2.0/Database/PQF_label/ground_truth/train_108/QP"
    dir_save_stack_pre = "/home/x/SCI_1/MFQEv2.0/Database/PQF_enhancement/QP"

    PQFLabel_sub = "_MaxNfs_300.hdf5"

    batch_size = 64
    max_batch_PerNpy = 14500
    ratio_training = 1.0 # we select a small part of test set for validation
    max_NonPQF_OneVideo = 20

    main_extract_TrainingSet()

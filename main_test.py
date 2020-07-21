import glob, os
import numpy as np
import tensorflow as tf
from skimage.measure import compare_psnr, compare_ssim
import net_MFCNN

# CHANGE YOUR INFO!
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # gpu
QP_video = 37  # QP
opt_out = False  # store and output enhanced frames

dir_CmpVideo = "data/test/compressed"
dir_RawVideo = "data/test/raw"
dir_PQFLabel = "data/PQF_label/estimated/test_18/QP" + str(QP_video)
dir_model = "model"
dir_out = "out"
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
file_object = open(os.path.join(dir_out, "record_test.txt"), 'w')

opt_QPLabel = False  # optional. see README
dir_ApprQP = "data/PQF_label"

QP_list = [22,27,32,37,42]
net1_list = [37,42]  # network1 for QP37 and 42, network2 for other QPs. see README

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only show error and warning
config = tf.ConfigProto(allow_soft_placement = True)  # if GPU is not usable, then turn to CPU automatically

BATCH_SIZE = 1
CHANNEL = 1

# search and test all cmp videos
CmpVideo_path_list = glob.glob(os.path.join(dir_CmpVideo, "*.yuv"))
num_CmpVideo = len(CmpVideo_path_list)


def y_import(video_path, height_frame, width_frame, nfs, startfrm):
    """Import Y channel from a yuv video.

    startfrm: start from 0
    return: (nfs * height * width), dtype=uint8."""

    d0 = height_frame // 2
    d1 = width_frame // 2
    y_size = height_frame * width_frame
    u_size = d0 * d1
    v_size = u_size

    fp = open(video_path,'rb')

    # target at startfrm
    blk_size = y_size + u_size + v_size
    fp.seek(blk_size * startfrm, 0)

    # extract
    y_batch = []
    for ite_frame in range(nfs):
        y_frame = [ord(fp.read(1)) for k in range(y_size)]
        y_frame = np.array(y_frame, dtype=np.uint8).reshape((height_frame, width_frame))
        fp.read(u_size + v_size)  # skip u and v
        y_batch.append(y_frame)
    fp.close()
    y_batch = np.array(y_batch)
    return y_batch


def return_PQFIndices(PQF_label, QP, ApprQP_label):
    """find all PQFs and their pre/sub PQFs pertain to this QP."""
    PQF_indices = [i for i in range(len(PQF_label)) if PQF_label[i] == 1]
    
    ApprQPLabel_PQF = [ApprQP_label[i] for i in range(len(ApprQP_label)) if i in PQF_indices]

    PQF_order_part = [o for o in range(len(ApprQPLabel_PQF)) if ApprQPLabel_PQF[o] == QP]
    PQFIndex_list_part = [PQF_indices[o] for o in range(len(PQF_indices)) if o in PQF_order_part]
    
    if len(PQFIndex_list_part) == 0:
        return [],[],[]
        
    num_PQF = len(PQFIndex_list_part)
    
    CmpPQFIndex_list_part = PQFIndex_list_part.copy()
    PrePQFIndex_list_part = PQFIndex_list_part[0: (num_PQF - 1)]
    SubPQFIndex_list_part = PQFIndex_list_part[1: num_PQF]
    
    PrePQFIndex_list_part = [PQFIndex_list_part[0]] + PrePQFIndex_list_part
    SubPQFIndex_list_part.append(PQFIndex_list_part[-1])
    
    return PrePQFIndex_list_part, CmpPQFIndex_list_part, SubPQFIndex_list_part
    

def return_NPIndices(PQF_label, QP, ApprQP_label):
    """find all non-PQFs and their pre/sub PQFs pertain to this QP."""
    PQFIndex_list = [i for i in range(len(PQF_label)) if PQF_label[i] == 1]

    # find unqualified non-PQFs and their sub PQFs. Pre PQFs are themselves.
    NonPQFIndex_list = [i for i in range(len(PQF_label)) if (PQF_label[i] == 0) and (i < PQFIndex_list[0])]
    PrePQFIndex_list = NonPQFIndex_list.copy()
    SubPQFIndex_list = [PQFIndex_list[0]] * len(NonPQFIndex_list)
    
    # find qualified non-PQFs and their pre/sub PQFs.
    NonPQFIndex_list_good = [i for i in range(len(PQF_label)) if (PQF_label[i] == 0) and (i > PQFIndex_list[0]) and (i < PQFIndex_list[-1])]
    NonPQFIndex_list += NonPQFIndex_list_good
    num_NonPQF = len(NonPQFIndex_list_good)
    for ite_NonPQF in range(num_NonPQF):
        
        index_NonPQF = NonPQFIndex_list_good[ite_NonPQF]
        
        for ite_PQF in range(len(PQFIndex_list) - 1):
            
            if (PQFIndex_list[ite_PQF] < index_NonPQF) and (PQFIndex_list[ite_PQF + 1] > index_NonPQF):
            
                PrePQFIndex_list.append(PQFIndex_list[ite_PQF])
                SubPQFIndex_list.append(PQFIndex_list[ite_PQF + 1])
                break
                
    # find unqualified non-PQFs and their sub PQFs. Sub PQFs are themselves.
    NonPQFIndex_list_bad = [i for i in range(len(PQF_label)) if (PQF_label[i] == 0) and (i > PQFIndex_list[-1])]
    NonPQFIndex_list += NonPQFIndex_list_bad
    PrePQFIndex_list += [PQFIndex_list[-1]] * len(NonPQFIndex_list_bad)
    SubPQFIndex_list += NonPQFIndex_list_bad
          
    # find non-PQFs pertain to this QP      
    ApprQPLabel_nonPQF = [ApprQP_label[i] for i in range(len(ApprQP_label)) if i in NonPQFIndex_list]

    NonPQF_order_part = [o for o in range(len(ApprQPLabel_nonPQF)) if ApprQPLabel_nonPQF[o] == QP]
    NonPQFIndex_list_part = [NonPQFIndex_list[o] for o in range(len(NonPQFIndex_list)) if o in NonPQF_order_part]

    if len(NonPQFIndex_list_part) == 0:
        return [],[],[]
    
    PrePQFIndex_list_part = [PrePQFIndex_list[o] for o in range(len(PrePQFIndex_list)) if o in NonPQF_order_part]
    SubPQFIndex_list_part = [SubPQFIndex_list[o] for o in range(len(SubPQFIndex_list)) if o in NonPQF_order_part]   
                
    return PrePQFIndex_list_part, NonPQFIndex_list_part, SubPQFIndex_list_part


def isplane(frame):
    """detect black frames or other plane frames."""
    tmp_array = np.squeeze(frame).reshape([-1])

    if all(tmp_array[1:] == tmp_array[:-1]): # all values in this frame are equal
        return True
    else:
        return False


def func_enhance(dir_model_pre, QP, PreIndex_list, CmpIndex_list, SubIndex_list):
    """enhance PQFs or non-PQFs
    update dpsnr, dssim and enhanced frames."""
    global sum_dpsnr, sum_dssim
    if opt_out:
        global enhanced_list
    
    tf.reset_default_graph()  # reset graph for new video input

    # defind enhancement process
    x1 = tf.placeholder(tf.float32, [BATCH_SIZE, height, width, CHANNEL])  # previous
    x2 = tf.placeholder(tf.float32, [BATCH_SIZE, height, width, CHANNEL])  # current
    x3 = tf.placeholder(tf.float32, [BATCH_SIZE, height, width, CHANNEL])  # subsequent
        
    if QP in net1_list:
        is_training = tf.placeholder_with_default(False, shape=())
    
    x1to2 = net_MFCNN.warp_img(BATCH_SIZE, x2, x1, False)
    x3to2 = net_MFCNN.warp_img(BATCH_SIZE, x2, x3, True)
    
    if QP in net1_list:
        x2_enhanced = net_MFCNN.network(x1to2, x2, x3to2, is_training)
    else:
        x2_enhanced = net_MFCNN.network2(x1to2, x2, x3to2)
    
    saver = tf.train.Saver()
    
    with tf.Session(config = config) as sess:

        # restore model
        model_path = os.path.join(dir_model_pre, "model_step2.ckpt-" + str(QP))
        saver.restore(sess, model_path)
    
        nfs = len(CmpIndex_list)
        
        sum_dpsnr_part = 0.0
        sum_dssim_part = 0.0
    
        for ite_frame in range(nfs):
          
            # load frames
            pre_frame = y_import(CmpVideo_path, height, width, 1, PreIndex_list[ite_frame])[:,:,:,np.newaxis] / 255.0
            cmp_frame = y_import(CmpVideo_path, height, width, 1, CmpIndex_list[ite_frame])[:,:,:,np.newaxis] / 255.0
            sub_frame = y_import(CmpVideo_path, height, width, 1, SubIndex_list[ite_frame])[:,:,:,np.newaxis] / 255.0
            
            # if cmp frame is plane?
            if isplane(cmp_frame):
                continue

            # if PQF frames are plane?
            if isplane(pre_frame):
                 pre_frame = np.copy(cmp_frame)
            if isplane(sub_frame):
                 sub_frame = np.copy(cmp_frame)
  
            # enhance
            if QP in net1_list:
                enhanced_frame = sess.run(x2_enhanced, feed_dict={x1:pre_frame, x2:cmp_frame, x3:sub_frame, is_training:False})
            else:
                enhanced_frame = sess.run(x2_enhanced, feed_dict={x1:pre_frame, x2:cmp_frame, x3:sub_frame})
                
            # record for output video
            if opt_out:
                enhanced_list[CmpIndex_list[ite_frame]] = np.squeeze(enhanced_frame)
            
            # evaluate and accumulate dpsnr
            raw_frame = np.squeeze(y_import(RawVideo_path, height, width, 1, CmpIndex_list[ite_frame])) / 255.0
            cmp_frame = np.squeeze(cmp_frame)
            enhanced_frame = np.squeeze(enhanced_frame)
            
            raw_frame = np.float32(raw_frame)
            cmp_frame = np.float32(cmp_frame)
            
            psnr_ori = compare_psnr(cmp_frame, raw_frame, data_range=1.0)
            psnr_aft = compare_psnr(enhanced_frame, raw_frame, data_range=1.0)

            ssim_ori = compare_ssim(cmp_frame, raw_frame, data_range=1.0)
            ssim_aft = compare_ssim(enhanced_frame, raw_frame, data_range=1.0)
            
            sum_dpsnr_part += psnr_aft - psnr_ori
            sum_dssim_part += ssim_aft - ssim_ori
            
            print("%d | %d at QP = %d" % (ite_frame + 1, nfs, QP), end="\r")
        print(" "*20, end="\r")
        
        sum_dpsnr += sum_dpsnr_part
        sum_dssim += sum_dssim_part
        
        average_dpsnr = sum_dpsnr_part / nfs
        average_dssim = sum_dssim_part / nfs
        print("dPSNR: %.3f - dSSIM: %.3f - nfs: %4d" % (average_dpsnr, average_dssim, nfs), flush=True)
        file_object.write("dPSNR: %.3f - dSSIM: %.3f - nfs: %4d\n" % (average_dpsnr, average_dssim, nfs))
        file_object.flush()


# enhancement video by video
for ite_CmpVideo in range(num_CmpVideo):
    
    # extract info from cmp video path
    CmpVideo_path = CmpVideo_path_list[ite_CmpVideo]
    CmpVideo_name = os.path.basename(CmpVideo_path).split(".")[0]
    
    RawVideo_name = CmpVideo_name
    RawVideo_path = os.path.join(dir_RawVideo, RawVideo_name + ".yuv")
    
    nfs = int(CmpVideo_name.split("_")[2])
    dims_list = CmpVideo_name.split("_")[1]
    width = int(dims_list.split("x")[0])
    height = int(dims_list.split("x")[1])
    
    # load PQF label and ApprQP label
    PQF_label = list(np.load(os.path.join(dir_PQFLabel, "PQFLabel_" + CmpVideo_name + ".npy")))
    if opt_QPLabel:
        ApprQP_label = list(np.load(os.path.join(dir_ApprQP, "ApprQP_" + CmpVideo_name + ".npy")))
    else:
        ApprQP_label = [QP_video] * nfs
    
    # initialize enhanced_list
    if opt_out:
        enhanced_list = []
    
    # record dpsnr and dssim
    sum_dpsnr = 0.0
    sum_dssim = 0.0

    # enhance PQF
    print("enhancing PQF...")
    for QP in QP_list:
    
        # find all PQFs and their pre/sub PQFs pertain to this QP
        PrePQFIndex_list_part, CmpPQFIndex_list_part, SubPQFIndex_list_part = return_PQFIndices(PQF_label, QP, ApprQP_label)
        if len(PrePQFIndex_list_part) == 0:
            continue
            
        # enhance PQF
        dir_model_pre = os.path.join(dir_model, "P_enhancement")
        func_enhance(dir_model_pre, QP, PrePQFIndex_list_part, CmpPQFIndex_list_part, SubPQFIndex_list_part)
        
    # enhance Non-PQF
    print("enhancing non-PQFs...")
    for QP in QP_list:
        
        # find pre-PQFs, non-PQFs and sub-PQFs pertain to this QP
        PrePQFIndex_list_part, NonPQFIndex_list_part, SubPQFIndex_list_part = return_NPIndices(PQF_label, QP, ApprQP_label)
        if len(PrePQFIndex_list_part) == 0:
            continue
                    
        # enhance non-PQF
        dir_model_pre = os.path.join(dir_model, "NP_enhancement")
        func_enhance(dir_model_pre, QP, PrePQFIndex_list_part, NonPQFIndex_list_part, SubPQFIndex_list_part)

    # output and record result
    average_dpsnr = sum_dpsnr / nfs
    average_dssim = sum_dssim / nfs
    print("dPSNR: %.3f - dSSIM: %.3f - nfs: %4d - %s" % (average_dpsnr, average_dssim, nfs, CmpVideo_name), flush=True)
    file_object.write("dPSNR: %.3f - dSSIM: %.3f - nfs: %4d - %s\n" % (average_dpsnr, average_dssim, nfs, CmpVideo_name))
    file_object.flush()
    
    # save bmp
    # here we have enhanced_list that records all enhanced frames. If you want to output enhanced images, code here.
    if opt_out:
        enhanced_list = np.array(enhanced_list, dtype=np.float32)
        pass

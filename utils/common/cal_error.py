import numpy as np

def cal_error(radar, trajectory_info, frame_index, ospa_error_set, target_set_num):
    # 计算 OSPA 与 目标的势
    legacy_PT_num = 0
    if radar.legacy_PT_num > 0:
        groud_truth = trajectory_info['labels_set'][frame_index][1:,:]
        track_estimation = []
        for track_index in range(radar.legacy_PT_num):
            if len(radar.legacy_PT_set[track_index].X) >= radar.confirm_trajectory_len:
                track_estimation.append(radar.legacy_PT_set[track_index].X[-1])
                legacy_PT_num += 1
        if len(track_estimation) > 0:
            track_estimation = np.hstack(track_estimation)
        else:
            track_estimation = np.zeros((radar.x_dimension,1))
        ospa_error_set.append(radar.ospa(track_estimation.T, groud_truth.T, c=100, p=1))
    else:
        ospa_error_set.append(0)

    target_set_num.append(legacy_PT_num)


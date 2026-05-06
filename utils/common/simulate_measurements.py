import numpy as np
from utils.common.plot_track import *
'''
功能：产生当前时刻的量测
'''
def simulate_measurements(radar, measurement_plots_each_frame, prob_detection=1.0):
    plot_track_set = []
    for measurement_index in range(measurement_plots_each_frame.shape[1]): # 遍历每一个点迹
        if prob_detection >= np.random.random(1):
            # 该点迹被检测到了
            plot_track_set.append(Plot_Track(measurement_plots_each_frame[:, measurement_index],measurement_index))

    radar.plot_track_set = plot_track_set
    radar.plot_track_num = len(plot_track_set)

    return radar
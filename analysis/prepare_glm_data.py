"""
GLM Data Preparation Script

This script processes raw behavioral and imaging data to create glm_data_dict.pkl.gz files
that are used for GLM encoding model analysis. It supports both unisensory (AVC) and 
multisensory (VACANT) task types.

Usage:
    python prepare_glm_data.py --task avc --base-path /path/to/data
    python prepare_glm_data.py --task priming --base-path /path/to/data
    python prepare_glm_data.py --task vacant --base-path /path/to/data
    python prepare_glm_data.py --task all --base-path /path/to/data
"""

import argparse
import gzip
import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# Session Registries
# ============================================================================

# PPC_AVC = {
#     'IS-2-1L': ['2025-03-10', '2025-03-11', '2025-03-12', '2025-03-13',
#                 '2025-03-20', '2025-03-26', '2025-04-02', '2025-04-04',
#                 '2025-04-08', '2025-04-09', '2025-05-05',
#                 '2025-05-28', '2025-05-30', '2025-11-20'],
#     'IS-3-1R': ['2024-12-23', '2024-12-24'],
#     'JP-2-1L': ['2025-08-19', '2025-08-26', '2025-09-16', '2025-09-17',
#                 '2025-09-19', '2025-09-22'],
#     'KA-1-00': ['2025-12-16', '2026-01-08'],
#     'KA-2-00': ['2026-01-08', '2026-01-13', '2026-01-14'],
#     'KO-5-1L': ['2026-03-05', '2026-03-06', '2026-03-12', '2026-03-13', 
#                 '2026-03-24', '2026-03-30', '2026-04-06', '2026-04-27'],
#     'KO-6-00': ['2026-03-12', '2026-03-23', '2026-03-25', '2026-03-30', '2026-04-06'],
# }

PPC_AVC_PRIMING = {
    'IS-2-1L': ['2025-04-14', '2025-04-15', '2025-04-22', '2025-12-12'],
    'KO-5-1L': ['2026-04-01', '2026-04-02', '2026-04-08', '2026-04-28',
                '2026-04-29', '2026-04-30'],
    'KO-6-00': ['2026-04-02', '2026-04-07']
}

PPC_VACANT = {
    'IS-2-1L': ['2025-04-14_1', '2025-04-15_1', '2025-04-22_1', '2025-12-12_1'],
    'KO-5-1L': ['2026-04-01_1', '2026-04-02_1', '2026-04-08_1', '2026-04-28_1',
                '2026-04-29_1', '2026-04-30_1'],
    'KO-6-00': ['2026-04-02_1', '2026-04-07_1']
}


# ============================================================================
# TrialDataHandler Class
# ============================================================================

class TrialDataHandler:
    """
    Post-alignment trial and behavioral data processing.
    Encapsulates imaged-trial extraction, behavior/task DataFrames,
    neural/movement extraction, and task-period activity windows.
    """

    # Class-level mapping dictionaries (shared across instances)
    visual_map = {
        'psych': {
            1: 0, 2: 15, 3: 25, 4: 65, 5: 75, 6: 90,
            7: 0, 8: 15, 9: 25, 10: 65, 11: 75, 12: 90,
            13: 0, 14: 15, 15: 25, 16: 65, 17: 75, 18: 90
        },
        'vis_psych': {1: 0, 2: 15, 3: 30, 4: 60, 5: 75, 6: 90, 7: 45},
        'vis_psych_hard': {1: 0, 2: 15, 3: 30, 4: 40, 5: 50, 6: 60, 7: 75, 8: 90, 9: 45},
        'cong_psych_hard': {1: 0, 2: 15, 3: 30, 4: 40, 5: 50, 6: 60, 7: 75, 8: 90, 9: 45},
        '2loc': {1: 0, 2: 90, 3: 0, 4: 90, 5: 90, 6: 0},
        'avc': {1: 0, 2: 90, 3: 0, 4: 90, 5: np.nan, 6: np.nan},
        'aud': {1: np.nan, 2: np.nan},
        'MIT': {1: 0, 2: 1},
        'MIT_occluder': {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1}
    }

    audio_map = {
        'psych': {
            1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1,
            7: 1, 8: 1, 9: 1, 10: 0, 11: 0, 12: 0,
            13: 1, 14: 1, 15: 1, 16: 0, 17: 0, 18: 0
        },
        'vis_psych': {1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
        'vis_psych_hard': {1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan},
        'cong_psych_hard': {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: np.nan},
        '2loc': {1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1},
        'avc': {1: 0, 2: 1, 3: np.nan, 4: np.nan, 5: 0, 6: 1},
        'aud': {1: 0, 2: 1},
        'MIT': {1: np.nan, 2: np.nan},
        'MIT_occluder': {1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan}
    }

    context_map = {
        'psych': {
            1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
            7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1,
            13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2
        },
        'vis_psych': {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
        'vis_psych_hard': {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
        'cong_psych_hard': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: np.nan},
        '2loc': {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2},
        'avc': {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2},
        'aud': {1: 2, 2: 2},
        'MIT': {1: 0, 2: 0},
        'MIT_occluder': {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    }

    def __init__(self, tasktype='psych'):
        self.tasktype = tasktype
        self.data_dict = None
        self.imaged_trials = None
        self.task_df = None

    def get_imaged_trials(self, imaging, tasktype=None):
        tasktype = tasktype if tasktype is not None else self.tasktype
        imaged_trials = []
        for trial_key in imaging.keys():
            if not trial_key.startswith('trial_'):
                continue
            trial_num = int(trial_key.split('_')[1])
            trial_data = imaging[trial_key]
            if ('dff' in trial_data and isinstance(trial_data['dff'], (np.ndarray, list)) and trial_data['dff'].size > 0) or \
               ('deconv' in trial_data and isinstance(trial_data['deconv'], (np.ndarray, list)) and trial_data['deconv'].size > 0):
                imaged_trials.append(trial_num)
        imaged_trials = sorted(imaged_trials)

        data_dict = {}
        dff_data_dict = {}
        deconv_data_dict = {}
        movement_data_dict = {}
        behavior_data_dict = {}
        dff_shapes_dict = {}

        for trial_key in imaging.keys():
            if not trial_key.startswith('trial_'):
                continue
            trial_num = trial_key.split('_')[1]
            trial_data = imaging[trial_key]
            if 'dff' not in trial_data or 'deconv' not in trial_data:
                continue
            if isinstance(trial_data['dff'], (np.ndarray, list)):
                trial_shape = trial_data['dff'].shape[1]
            else:
                continue
            if trial_shape < 100 or trial_shape > 500:
                continue
            dff_data_dict[trial_num] = trial_data['dff']
            deconv_data_dict[trial_num] = trial_data['deconv']
            movement_data_dict[trial_num] = trial_data['movement_in_imaging_time']
            behavior_data_dict[trial_num] = trial_data['virmen_trial_info']
            dff_shapes_dict[trial_num] = trial_shape
            if 'virmen_trial_info' not in trial_data:
                continue

            trial_dict = {}
            if tasktype == 'MIT_occluder':
                mov_features = ['x_position', 'y_position', 'view_angle', 'mvData_1', 'mvData_2', 'mvData_3', 'cuePos', 'is_reward', 'in_ITI', 'ball_center']
                mov_info = trial_data['movement_in_imaging_time']
                for feature in mov_features:
                    trial_dict[feature] = np.array(mov_info[feature])
            else:
                mov_features = ['y_position', 'x_position', 'y_velocity', 'x_velocity', 'view_angle', 'is_reward', 'in_ITI', 'pure_tones']
                mov_info = trial_data['movement_in_imaging_time']
                for feature in mov_features:
                    trial_dict[feature] = np.array(mov_info[feature])
            behav_features = ['correct', 'left_turn', 'condition']
            behav_info = trial_data['virmen_trial_info']
            for feature in behav_features:
                if feature in behav_info:
                    trial_dict[feature] = int(behav_info[feature])
            neural_features = ['deconv', 'dff', 'z_dff']
            for feature in neural_features:
                trial_dict[feature] = np.array(trial_data[feature])
            required_features = behav_features
            if all(feature in trial_dict for feature in required_features):
                data_dict[trial_num] = trial_dict

        self.data_dict = data_dict
        self.imaged_trials = imaged_trials
        return data_dict, imaged_trials

    def create_behavior_df(self, data_dict=None, imaged_trials=None, tasktype=None):
        data_dict = data_dict if data_dict is not None else self.data_dict
        tasktype = tasktype if tasktype is not None else self.tasktype
        trial_data = []
        for trial_num in data_dict:
            trial_info = data_dict[trial_num]
            trial_data.append({
                'Trial': int(trial_num), 
                'trial_type': trial_info['condition'],
                'choice': trial_info['left_turn'],
                'outcome': trial_info['correct']
            })
        df = pd.DataFrame(trial_data)
        df['Visual_Stim'] = df['trial_type'].map(self.visual_map[tasktype])
        df['Audio_Stim'] = df['trial_type'].map(self.audio_map[tasktype])
        df['context'] = df['trial_type'].map(self.context_map[tasktype])
        return df

    def task_event_frames(self, data_dict=None, df=None):
        data_dict = data_dict if data_dict is not None else self.data_dict
        if df is None:
            raise ValueError("task_event_frames requires df (or set self.task_df first).")
        sound_onset = []
        iti_start = []
        trial_length = []
        for trial in data_dict.keys():
            y_pos = data_dict[trial]['y_position']
            sound_onset_frame = np.where(y_pos > 50)[0][0]
            iti = data_dict[trial]['in_ITI']
            sound_onset.append(sound_onset_frame)
            iti_start.append(np.where(iti > 0)[0][0])
            trial_length.append(len(y_pos))
        df = df.copy()
        df['sound onset'] = sound_onset
        df['iti start'] = iti_start
        df['trial length'] = trial_length
        return df

    def prune_task_df(self, task_df=None):
        df_in = task_df if task_df is not None else self.task_df
        if df_in is None:
            raise ValueError("prune_task_df requires task_df (or set self.task_df first).")
        df = df_in.copy()
        df['prev_choice'] = df['choice'].shift(1)
        df.iloc[0, df.columns.get_loc('prev_choice')] = 0
        df['iti start'] = df['iti start']
        mask = (df['trial length'] > 120) & (df['trial length'] < 500)
        df = df[mask]
        mask = df['trial length'] - df['iti start'] >= 50
        df = df[mask]
        df['choice'] = 1 - df['choice']
        df['prev_choice'] = 1 - df['prev_choice']
        df['choice_bin'] = df['choice'].map({0: -1, 1: 1})
        df['prev_choice_bin'] = df['prev_choice'].map({0: -1, 1: 1})
        df['outcome_bin'] = df['outcome'].map({0: -1, 1: 1})
        df['stim_binary'] = np.where(df['Visual_Stim'] == 90, 1, -1).astype(int)
        return df

    def neural_data_extraction(self, data_dict=None, imaged_trials=None):
        data_dict = data_dict if data_dict is not None else self.data_dict
        trial_keys = [t for t in data_dict]
        trial_data_dff = []
        trial_data_deconv = []
        trial_size_lenth = []
        for trial in trial_keys:
            neural_data_deconv = data_dict[trial]['deconv'].T.copy()
            neural_data_dff = data_dict[trial]['dff'].T
            neural_data_deconv[neural_data_deconv[:, :] < 0.05] = 0
            trial_data_deconv.append(neural_data_deconv)
            trial_data_dff.append(neural_data_dff)
            trial_size_lenth.append(neural_data_deconv.shape[0])
        return trial_data_dff, trial_data_deconv, trial_size_lenth

    def movement_data_extraction(self, data_dict=None, imaged_trials=None, tasktype=None):
        data_dict = data_dict if data_dict is not None else self.data_dict
        tasktype = tasktype if tasktype is not None else self.tasktype
        trial_keys = [t for t in data_dict]
        trial_data_movement = []
        if tasktype == 'MIT_occluder':
            for trial in trial_keys:
                expected_shape = (1, len(data_dict[trial]['dff'][0]))
                nan_array = np.full(expected_shape, np.nan)
                y_position = data_dict[trial].get('y_position', nan_array).T
                x_position = data_dict[trial].get('x_position', nan_array).T
                mvData_1 = data_dict[trial].get('mvData_1', nan_array).T
                mvData_2 = data_dict[trial].get('mvData_2', nan_array).T
                mvData_3 = data_dict[trial].get('mvData_3', nan_array).T
                view_angle = data_dict[trial].get('view_angle', nan_array).T
                trial_data_movement.append([y_position, x_position, mvData_1, mvData_2, mvData_3, view_angle])
        else:
            for trial in trial_keys:
                expected_shape = (1, len(data_dict[trial]['dff'][0]))
                nan_array = np.full(expected_shape, np.nan)
                y_position = data_dict[trial].get('y_position', nan_array).T
                x_position = data_dict[trial].get('x_position', nan_array).T
                y_velocity = data_dict[trial].get('y_velocity', nan_array).T
                x_velocity = data_dict[trial].get('x_velocity', nan_array).T
                view_angle = data_dict[trial].get('view_angle', nan_array).T
                trial_data_movement.append([y_position, x_position, y_velocity, x_velocity, view_angle])
        return trial_data_movement

    def extract_task_period_activity(self, neural_data, task_df_pruned):
        task_df_pruned = task_df_pruned.copy().reset_index()
        baseline_activity_all_trials = []
        stimulus_activity_all_trials = []
        choice_activity_all_trials = []
        iti_activity_all_trials = []
        task_windows_list = []
        BASELINE_SIZE = 15
        STIMULUS_SIZE = 45
        CHOICE_SIZE = 45
        ITI_SIZE = 74
        for trial_idx in range(len(task_df_pruned)):
            try:
                trial_data = neural_data[task_df_pruned.iloc[trial_idx].name]
                iti_start = task_df_pruned['iti start'][trial_idx]
                three_seconds_before_iti = iti_start - 45
                baseline_activity = trial_data[0:min(BASELINE_SIZE, trial_data.shape[0]), :]
                if baseline_activity.shape[0] < BASELINE_SIZE:
                    pad_size = BASELINE_SIZE - baseline_activity.shape[0]
                    baseline_activity = np.pad(baseline_activity, ((0, pad_size), (0, 0)), mode='edge')
                stim_end = min(15 + STIMULUS_SIZE, trial_data.shape[0])
                stimulus_period_activity = trial_data[15:stim_end, :]
                if stimulus_period_activity.shape[0] < STIMULUS_SIZE:
                    pad_size = STIMULUS_SIZE - stimulus_period_activity.shape[0]
                    stimulus_period_activity = np.pad(stimulus_period_activity, ((0, pad_size), (0, 0)), mode='edge')
                choice_start = max(0, min(three_seconds_before_iti, trial_data.shape[0] - 1))
                choice_end = min(iti_start, trial_data.shape[0])
                choice_period_activity = trial_data[choice_start:choice_end, :]
                if choice_period_activity.shape[0] < CHOICE_SIZE:
                    pad_size = CHOICE_SIZE - choice_period_activity.shape[0]
                    choice_period_activity = np.pad(choice_period_activity, ((0, pad_size), (0, 0)), mode='edge')
                elif choice_period_activity.shape[0] > CHOICE_SIZE:
                    choice_period_activity = choice_period_activity[-CHOICE_SIZE:, :]
                iti_start_idx = max(0, trial_data.shape[0] - ITI_SIZE - 1)
                iti_period_activity = trial_data[iti_start_idx:-1, :]
                if iti_period_activity.shape[0] < ITI_SIZE:
                    pad_size = ITI_SIZE - iti_period_activity.shape[0]
                    iti_period_activity = np.pad(iti_period_activity, ((0, pad_size), (0, 0)), mode='edge')
                elif iti_period_activity.shape[0] > ITI_SIZE:
                    iti_period_activity = iti_period_activity[-ITI_SIZE:, :]
                baseline_activity_all_trials.append(baseline_activity)
                stimulus_activity_all_trials.append(stimulus_period_activity)
                choice_activity_all_trials.append(choice_period_activity)
                iti_activity_all_trials.append(iti_period_activity)
                task_windows = [(0, BASELINE_SIZE), (10, 10 + STIMULUS_SIZE), (choice_start, choice_end), (iti_start_idx, iti_start_idx + ITI_SIZE)]
                task_windows_list.append(task_windows)
            except Exception as e:
                print(f"Warning for trial {trial_idx}: {str(e)}")
                n_neurons = neural_data[0].shape[1]
                baseline_activity_all_trials.append(np.zeros((BASELINE_SIZE, n_neurons)))
                stimulus_activity_all_trials.append(np.zeros((STIMULUS_SIZE, n_neurons)))
                choice_activity_all_trials.append(np.zeros((CHOICE_SIZE, n_neurons)))
                iti_activity_all_trials.append(np.zeros((ITI_SIZE, n_neurons)))
                task_windows_list.append([(0, BASELINE_SIZE), (10, 10 + STIMULUS_SIZE), (0, CHOICE_SIZE), (0, ITI_SIZE)])
        baseline_activity_all_trials = np.array(baseline_activity_all_trials)
        stimulus_activity_all_trials = np.array(stimulus_activity_all_trials)
        choice_activity_all_trials = np.array(choice_activity_all_trials)
        iti_activity_all_trials = np.array(iti_activity_all_trials)
        return (baseline_activity_all_trials, stimulus_activity_all_trials, choice_activity_all_trials, iti_activity_all_trials, task_windows_list), task_df_pruned

    def extract_task_period_movement(self, movement_data, task_df_pruned):
        task_df_pruned = task_df_pruned.reset_index(drop=True)
        movement_data_pruned = [movement_data[i] for i in range(len(task_df_pruned))]

        baseline_movement = []
        stimulus_movement = []
        choice_movement = []
        iti_movement = []

        for trial in range(len(task_df_pruned)):
            iti_start = task_df_pruned['iti start'][trial]
            baseline = [feature[0:15] for feature in movement_data_pruned[trial]]
            stimulus = [feature[15:60] for feature in movement_data_pruned[trial]]
            choice = [feature[iti_start - 45:iti_start] for feature in movement_data_pruned[trial]]
            iti = [feature[-75:-1] for feature in movement_data_pruned[trial]]
            baseline_movement.append(baseline)
            stimulus_movement.append(stimulus)
            choice_movement.append(choice)
            iti_movement.append(iti)

        baseline_movement = np.array(baseline_movement)
        stimulus_movement = np.array(stimulus_movement)
        choice_movement = np.array(choice_movement)
        iti_movement = np.array(iti_movement)
        return baseline_movement, stimulus_movement, choice_movement, iti_movement

    def build_glm_trials(self, data_dict=None, task_df_pruned=None, trial_data_dff=None,
                         trial_data_deconv=None, trial_data_movement=None, tasktype=None):
        """
        Build a list of trial objects with full-length neural and movement data.
        Only includes trials that appear in task_df_pruned (no warnings for dropped trials).

        Args:
            data_dict: Dictionary with trial data (keys are trial numbers as strings)
            task_df_pruned: DataFrame with behavioral data and event frames
            trial_data_dff: List of dff arrays, each (frames_i, n_neurons)
            trial_data_deconv: List of deconv arrays, each (frames_i, n_neurons)
            trial_data_movement: List of movement feature arrays
            tasktype: Task type string ('avc', 'MIT_occluder', etc.)

        Returns:
            glm_trials: List of dictionaries, one per pruned trial
        """
        data_dict = data_dict if data_dict is not None else self.data_dict
        tasktype = tasktype if tasktype is not None else self.tasktype

        if data_dict is None:
            raise ValueError("build_glm_trials requires data_dict (or set self.data_dict first).")
        if task_df_pruned is None:
            raise ValueError("build_glm_trials requires task_df_pruned.")
        if trial_data_dff is None:
            raise ValueError("build_glm_trials requires trial_data_dff.")
        if trial_data_deconv is None:
            raise ValueError("build_glm_trials requires trial_data_deconv.")
        if trial_data_movement is None:
            raise ValueError("build_glm_trials requires trial_data_movement.")

        glm_trials = []

        task_df_reset = task_df_pruned.copy().reset_index(drop=True)

        trial_keys = [t for t in data_dict]
        trial_num_to_list_idx = {int(t): i for i, t in enumerate(trial_keys)}

        for row_idx, row in task_df_reset.iterrows():
            trial_num = int(row["Trial"])
            if trial_num not in trial_num_to_list_idx:
                continue
            list_idx = trial_num_to_list_idx[trial_num]
            trial_key = trial_keys[list_idx]

            neural_dff = trial_data_dff[list_idx]
            neural_deconv = trial_data_deconv[list_idx]

            neural_z_dff = None
            if "z_dff" in data_dict[trial_key]:
                neural_z_dff = data_dict[trial_key]["z_dff"].T

            movement_feats = trial_data_movement[list_idx]

            if tasktype == "MIT_occluder":
                y_position, x_position, mvData_1, mvData_2, mvData_3, view_angle = movement_feats
                y_position = y_position.flatten() if y_position.ndim > 1 else y_position
                x_position = x_position.flatten() if x_position.ndim > 1 else x_position
                mvData_1 = mvData_1.flatten() if mvData_1.ndim > 1 else mvData_1
                mvData_2 = mvData_2.flatten() if mvData_2.ndim > 1 else mvData_2
                mvData_3 = mvData_3.flatten() if mvData_3.ndim > 1 else mvData_3
                view_angle = view_angle.flatten() if view_angle.ndim > 1 else view_angle
                movement_dict = {
                    "y_position": y_position,
                    "x_position": x_position,
                    "mvData_1": mvData_1,
                    "mvData_2": mvData_2,
                    "mvData_3": mvData_3,
                    "view_angle": view_angle,
                }
            else:
                y_position, x_position, y_velocity, x_velocity, view_angle = movement_feats
                y_position = y_position.flatten() if y_position.ndim > 1 else y_position
                x_position = x_position.flatten() if x_position.ndim > 1 else x_position
                y_velocity = y_velocity.flatten() if y_velocity.ndim > 1 else y_velocity
                x_velocity = x_velocity.flatten() if x_velocity.ndim > 1 else x_velocity
                view_angle = view_angle.flatten() if view_angle.ndim > 1 else view_angle
                movement_dict = {
                    "y_position": y_position,
                    "x_position": x_position,
                    "y_velocity": y_velocity,
                    "x_velocity": x_velocity,
                    "view_angle": view_angle,
                }

            sound_onset = int(row["sound onset"])
            iti_start = int(row["iti start"])
            trial_len = int(row["trial length"])

            neural_dict = {
                "dff": neural_dff,
                "deconv": neural_deconv,
            }
            if neural_z_dff is not None:
                neural_dict["z_dff"] = neural_z_dff

            trial_obj = {
                "trial_id": trial_num,
                "context": int(row["context"]),
                "choice": int(row["choice"]),
                "outcome": int(row["outcome"]),
                "trial_type": int(row["trial_type"]),
                "Visual_Stim": float(row["Visual_Stim"]) if not pd.isna(row["Visual_Stim"]) else np.nan,
                "Audio_Stim": float(row["Audio_Stim"]) if not pd.isna(row["Audio_Stim"]) else np.nan,
                "neural": neural_dict,
                "movement": movement_dict,
                "events": {
                    "sound_onset": sound_onset,
                    "iti_start": iti_start,
                    "trial_length": trial_len,
                },
            }
            glm_trials.append(trial_obj)

        return glm_trials


# ============================================================================
# Processing Functions
# ============================================================================

def _default_workers():
    n = cpu_count() or 4
    return max(1, min(n - 1, 8))


def process_one_session(args):
    """
    Run the full data pipeline for a single session.

    Loads imaging_data.pkl, builds data_dict and task_df_pruned, extracts
    neural/movement windows and GLM trials, and saves all outputs to the
    session directory.

    Parameters
    ----------
    args : tuple
        (mouse_ID : str, date : str, tasktype : str, base_path : str)

    Returns
    -------
    tuple
        (session_id : str, error : str | None)
        On success error is None. On failure error is the exception message.
    """
    mouse_ID, date, tasktype, base_path = args
    session_id = f"{mouse_ID}_{date}"
    session_dir = Path(base_path) / mouse_ID / date
    imaging_path = session_dir / "imaging_data.pkl"

    if not imaging_path.exists():
        return (session_id, "File not found")

    try:
        handler = TrialDataHandler(tasktype=tasktype)

        with open(imaging_path, "rb") as f:
            imaging_new = pickle.load(f)

        data_dict, imaged_trials = handler.get_imaged_trials(imaging_new, tasktype)
        if not data_dict:
            return (session_id, "No valid trials after get_imaged_trials")

        task_df = handler.create_behavior_df(data_dict=data_dict, tasktype=tasktype)
        task_df = handler.task_event_frames(data_dict=data_dict, df=task_df)
        task_df_pruned = handler.prune_task_df(task_df=task_df)
        if len(task_df_pruned) == 0:
            return (session_id, "No trials remaining after pruning")

        task_df_pruned.to_csv(session_dir / "task_df_pruned.csv", index=False)
        task_df_pruned.to_pickle(session_dir / "task_df_pruned.pkl")

        trial_data_dff, trial_data_deconv, _ = handler.neural_data_extraction(data_dict=data_dict)
        trial_data_movement = handler.movement_data_extraction(data_dict=data_dict, tasktype=tasktype)

        glm_trials = handler.build_glm_trials(
            data_dict=data_dict,
            task_df_pruned=task_df_pruned,
            trial_data_dff=trial_data_dff,
            trial_data_deconv=trial_data_deconv,
            trial_data_movement=trial_data_movement,
            tasktype=tasktype,
        )
        with gzip.open(session_dir / "glm_data_dict.pkl.gz", "wb") as f:
            pickle.dump(glm_trials, f, protocol=pickle.HIGHEST_PROTOCOL)

        trial_keys = list(data_dict.keys())
        trial_to_idx = {int(t): i for i, t in enumerate(trial_keys)}
        pruned_trial_nums = task_df_pruned["Trial"].values
        pruned_dff = [trial_data_dff[trial_to_idx[t]] for t in pruned_trial_nums if t in trial_to_idx]
        pruned_deconv = [trial_data_deconv[trial_to_idx[t]] for t in pruned_trial_nums if t in trial_to_idx]
        pruned_movement = [trial_data_movement[trial_to_idx[t]] for t in pruned_trial_nums if t in trial_to_idx]

        (baseline_dff, stimulus_dff, choice_dff, iti_dff, _), _ = handler.extract_task_period_activity(
            pruned_dff, task_df_pruned
        )
        (baseline_deconv, stimulus_deconv, choice_deconv, iti_deconv, _), _ = handler.extract_task_period_activity(
            pruned_deconv, task_df_pruned
        )
        baseline_mov, stimulus_mov, choice_mov, iti_mov = handler.extract_task_period_movement(
            pruned_movement, task_df_pruned
        )

        np.savez(
            session_dir / "neural_activity_windows_deconv.npz",
            baseline=baseline_deconv,
            stimulus=stimulus_deconv,
            choice=choice_deconv,
            iti=iti_deconv,
        )
        np.savez(
            session_dir / "neural_activity_windows_dff.npz",
            baseline=baseline_dff,
            stimulus=stimulus_dff,
            choice=choice_dff,
            iti=iti_dff,
        )
        np.savez(
            session_dir / "movement_windows.npz",
            baseline=baseline_mov,
            stimulus=stimulus_mov,
            choice=choice_mov,
            iti=iti_mov,
        )

        return (session_id, None)

    except Exception as e:
        return (session_id, str(e))


def run_parallel_sessions(ppc_avc, base_path, tasktype="avc", n_workers=None):
    """
    Process all sessions in ppc_avc in parallel.

    Parameters
    ----------
    ppc_avc : dict
        Map mouse_ID -> list of date strings (e.g. PPC_AVC).
    base_path : str
        Root directory for data: base_path / mouse_ID / date / ...
    tasktype : str
        Task type passed to TrialDataHandler (e.g. 'avc', 'psych').
    n_workers : int, optional
        Number of processes. Defaults to min(8, cpu_count() - 1).

    Returns
    -------
    tuple
        (success_list : list of str, fail_list : list of tuple)
        success_list = session_ids that completed without error.
        fail_list = [(session_id, error_message), ...].
    """
    task_list = []
    for mouse_ID, dates in ppc_avc.items():
        for date in dates:
            task_list.append((mouse_ID, date, tasktype, base_path))

    n_workers = n_workers or _default_workers()
    n_workers = min(n_workers, len(task_list)) if task_list else 0
    if n_workers <= 0:
        return ([], [])

    success_list = []
    fail_list = []
    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_one_session, task_list),
                total=len(task_list),
                desc="Sessions",
                unit="session",
            )
        )
    for session_id, err in results:
        if err is None:
            success_list.append(session_id)
        else:
            fail_list.append((session_id, err))

    return success_list, fail_list


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare GLM data for unisensory and multisensory tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_glm_data.py --task avc --base-path /Volumes/Data/ProcessedData
  python prepare_glm_data.py --task priming --base-path /Volumes/Data/ProcessedData
  python prepare_glm_data.py --task vacant --base-path /Volumes/Data/ProcessedData
  python prepare_glm_data.py --task all --base-path /Volumes/Data/ProcessedData --workers 4
        """
    )
    
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['avc', 'priming', 'vacant', 'all'],
        help="Which task to process: 'avc' (original AVC), 'priming' (AVC priming sessions), 'vacant' (multisensory sessions), or 'all'"
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        required=True,
        help="Base path to processed data directory (e.g., /Volumes/Data/ProcessedData)"
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    base_path = args.base_path
    n_workers = args.workers
    
    print("=" * 60)
    print("GLM Data Preparation")
    print("=" * 60)
    print(f"Base path: {base_path}")
    print(f"Task: {args.task}")
    print(f"Workers: {n_workers if n_workers else 'auto-detect'}")
    print()
    
    all_success = []
    all_failed = []
    
    if args.task in ['avc', 'all']:
        print("\n" + "=" * 60)
        print("Processing AVC (original) sessions")
        print("=" * 60)
        total_sessions = sum(len(dates) for dates in PPC_AVC.values())
        print(f"Total sessions: {total_sessions}")
        print()
        
        success, failed = run_parallel_sessions(
            PPC_AVC, base_path, tasktype='avc', n_workers=n_workers
        )
        all_success.extend(success)
        all_failed.extend(failed)
        
        print(f"\nAVC: Success: {len(success)} / {total_sessions}")
        if failed:
            print(f"AVC: Failed: {len(failed)}")
            for session_id, error in failed:
                print(f"  - {session_id}: {error}")
    
    if args.task in ['priming', 'all']:
        print("\n" + "=" * 60)
        print("Processing Priming (AVC) sessions")
        print("=" * 60)
        total_sessions = sum(len(dates) for dates in PPC_AVC_PRIMING.values())
        print(f"Total sessions: {total_sessions}")
        print()
        
        success, failed = run_parallel_sessions(
            PPC_AVC_PRIMING, base_path, tasktype='avc', n_workers=n_workers
        )
        all_success.extend(success)
        all_failed.extend(failed)
        
        print(f"\nPriming: Success: {len(success)} / {total_sessions}")
        if failed:
            print(f"Priming: Failed: {len(failed)}")
            for session_id, error in failed:
                print(f"  - {session_id}: {error}")
    
    if args.task in ['vacant', 'all']:
        print("\n" + "=" * 60)
        print("Processing VACANT (multisensory) sessions")
        print("=" * 60)
        total_sessions = sum(len(dates) for dates in PPC_VACANT.values())
        print(f"Total sessions: {total_sessions}")
        print()
        
        success, failed = run_parallel_sessions(
            PPC_VACANT, base_path, tasktype='2loc', n_workers=n_workers
        )
        all_success.extend(success)
        all_failed.extend(failed)
        
        print(f"\nVACANT: Success: {len(success)} / {total_sessions}")
        if failed:
            print(f"VACANT: Failed: {len(failed)}")
            for session_id, error in failed:
                print(f"  - {session_id}: {error}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Success: {len(all_success)}")
    print(f"Total Failed:  {len(all_failed)}")
    
    if all_failed:
        print("\nFailed sessions:")
        for session_id, error in all_failed:
            print(f"  - {session_id}: {error}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

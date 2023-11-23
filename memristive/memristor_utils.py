import numpy as np


def get_synaptic_weight(R, memristor_model):
    '''
    Mapping R to weights.
    Calculates the synaptic weight from the formula: alpha*(1.0 / R - 1 / Rc)
    :return: calculated synaptic weight
    '''

    return memristor_model.alpha * (1.0 / R - 1.0 / memristor_model.Rc)


def set_R_to_get_w(w, memristor_model):
    '''
    Mapping w to R.
    Calculates R needed to get w.
    :return: calculated R and w
    '''

    R = 1. / (1. / memristor_model.Rc + w / memristor_model.alpha)
    return R


def get_R_with_readout_noise(R, noise_level=0.01):
    '''
    Adding readout noise to R
    :return: R with readout noise
    '''
    noise = noise_level * R * np.random.uniform(-1., 1.)
    return R + noise


def generate_faults(n_weights, percentage_stuck, percentage_negative_change):
    idx = np.arange(n_weights)
    fault_factor = np.ones_like(idx, dtype=float)

    # Determine indices
    faults_stuck = np.random.choice(idx, int(percentage_stuck * n_weights), replace=False)
    print('Num. stuck memristors: ', faults_stuck.size)

    remaining_idx = np.setdiff1d(idx, faults_stuck, assume_unique=True)
    faults_negative_change = np.random.choice(remaining_idx, int(percentage_negative_change * n_weights), replace=False)
    print('Num. memristors with negative change: ', faults_negative_change.size)

    faults_positive_change = np.setdiff1d(remaining_idx, faults_negative_change, assume_unique=True)
    print('Num. memristors with positive change: ', faults_positive_change.size)

    assert faults_stuck.size + faults_positive_change.size + faults_negative_change.size == n_weights

    # Assign faults to previously (randomly) chosen indices
    # Note: Clipping range values are set manually here.
    fault_factor[faults_stuck] = 0.
    fault_factor[faults_negative_change] = np.clip(np.random.normal(-1., 0.5, size=faults_negative_change.size), -2,
                                                   -0.1)
    fault_factor[faults_positive_change] = np.clip(np.random.normal(1., 0.5, size=faults_positive_change.size), 0.1, 2)
    fault_factor = fault_factor.reshape((-1, 1))

    return fault_factor


def update_memristive_vars(memristor_model, w_new, mask_weights, R, noise_level, faults, is_last_iter):
    R_new = 1. / (1. / memristor_model.Rc + w_new / memristor_model.alpha)
    tmp_dR = R_new - R
    tmp_dR = mask_weights * tmp_dR

    faults_matrix = faults.reshape(tmp_dR.shape)
    dR = np.zeros_like(tmp_dR)
    dR_noisy = np.zeros_like(tmp_dR)

    # With noise
    # noise = noise_level * R * np.random.uniform(-1, 1, tmp_dR.shape)
    # Normal distribution for noise: mean = 0 (after subtracting R from it), std = noise_level * R / 3
    # Division by 3: because almost all data fall within 3 sigma
    noise = np.clip(np.random.normal(R, (noise_level * R) / 3, size=tmp_dR.shape),
                    R - noise_level * R,
                    R + noise_level * R) - R

    percent = 0.02  # for enforcing updates of certain magnitude (significant updates)
    if not is_last_iter:
        tup_idx = np.where(np.abs(tmp_dR) >= percent * R)
        # print(f'Above {percent * 100} percent: {tup_idx[0].size}')
    else:
        # In the last iter, enforce updates of 1% for the updates that are not significant
        # but those R with tmp_R of zero should not be enforced.
        tup_idx_1 = np.where(np.logical_and(tmp_dR > -percent * R, tmp_dR < 0))
        tmp_dR[tup_idx_1] = -percent / 2. * R[tup_idx_1]
        tup_idx_2 = np.where(np.logical_and(tmp_dR > 0, tmp_dR < percent * R))
        tmp_dR[tup_idx_2] = percent / 2. * R[tup_idx_2]

        tup_idx = np.where(tmp_dR != 0.0)  # both enforced and significant
        # assert tup_idx[0].size >= tup_idx_1[0].size + tup_idx_2[0].size, "error here"

    tmp_dR = mask_weights * tmp_dR
    dR[tup_idx] = tmp_dR[tup_idx]
    dR_noisy[tup_idx] = faults_matrix[tup_idx] * tmp_dR[tup_idx] + noise[tup_idx]
    R_new_noisy = R + dR_noisy
    w_new_noisy = get_synaptic_weight(R_new_noisy, memristor_model)

    # Update w_new there, where updates were significant
    w_new[tup_idx] = w_new_noisy[tup_idx]  # the rest is kept (previously calc. by optimizer)
    R = np.clip(R_new_noisy, memristor_model.R_min, memristor_model.R_max)

    return w_new, dR, noise, dR_noisy, R


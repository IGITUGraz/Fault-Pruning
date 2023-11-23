import os
from datetime import datetime
import yaml
import argparse
import copy

import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import regularizers
# import tensorflow_datasets as tfds

from sdict import sdict
from simmanager import SimManager
from simrecorder import Recorder, ZarrDataStore
from memristor_model import Memristor
from memristor_utils import get_synaptic_weight, set_R_to_get_w, generate_faults, update_memristive_vars
from pruning_tests import fit_line_no_intercept


def record_dict(prefix, d):
    for k, v in d.items():
        recorder.record('{}/{}'.format(prefix, k), np.array(v))


def prune(buffer, buffer_noisy, mask, threshold, stepwise_removal_percentage, min_connectivity,
           ffs, idx_flattened, history_window, use_fixed_w, weights, artificial_weights):
    """
    Do linear regression: y = ax + b (fit_line), or y = ax (fit_line_no_intercept)
    :param buffer: corresponds to x in linear regression
    :param buffer_noisy: corresponds to y in linear regression
    """
    # print('Sum mask weights before: ', np.sum(mask))
    a_arr = []
    i_arr = []
    for i in idx_flattened:
        if len(buffer[i]) >= history_window:
            # a, b = fit_line(np.array(buffer[i]), np.array(buffer_noisy[i]))
            a = fit_line_no_intercept(np.array(buffer[i]), np.array(buffer_noisy[i]))
            a_arr.append(a)
            i_arr.append(i)

    if len(a_arr) > 0:
        a_arr = np.array(a_arr)
        i_arr = np.array(i_arr)
        sorting_key = np.argsort(a_arr)
        key = np.where(a_arr[sorting_key] < threshold)

        removed = 0
        total = mask.size
        removed_wrong = 0

        for cnt, k in enumerate(i_arr[sorting_key[key]]):
            idx = np.unravel_index(k, mask.shape)

            f = ffs[k]
            if f > 0.5:
                removed_wrong += 1

            if mask[idx] == 1:
                if (removed + 1) / total > stepwise_removal_percentage:
                    continue
                if (np.sum(mask) - 1) / total < min_connectivity:
                    continue

                # remove element
                mask[idx] = 0
                removed += 1

                # add it to the artificial weights
                if use_fixed_w:
                    artificial_weights[idx] = weights[idx]

        if removed_wrong > 0:
            print(f'Removed: {removed}, removed_wrong {removed_wrong}.')
    return mask, artificial_weights


def main(config):
    inp_w = 28
    n_inp = inp_w * inp_w
    #################################
    # Prepare the training dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = np.reshape(x_train, (-1, n_inp)) / 255.
    x_test = np.reshape(x_test, (-1, n_inp)) / 255.

    # Reserve 10,000 samples for validation.
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(config.batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(config.batch_size)

    # Prepare the test dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(config.batch_size)

    #################################
    # Create the model
    inputs = keras.Input(shape=(n_inp,), name="digits")
    x1 = keras.layers.Dense(config.n_hidden, activation="relu")(inputs)
    outputs = keras.layers.Dense(config.n_out, activation="softmax", name="predictions")(x1)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Instantiate an optimizer
    initial_learning_rate = config.lr
    print('Cosine decay: ', config.lr_cosine_decay)
    if config.lr_cosine_decay:
        print("Optimizer will use the cosine decay learning rate with restarts.")
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            config.lr_decay_iter,
            t_mul=1.0,  # 2.0,
            m_mul=config.lr_decay,  # 1.0
            alpha=initial_learning_rate / 10,
        )
    else:  # Exponential decay
        print("Optimizer will use the exponential decay learning rate.")
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=config.lr_decay_iter,
            decay_rate=config.lr_decay,
            staircase=True)

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # Prepare the metrics.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    ######################################
    weights = model.get_weights()[::2]
    n_weights = np.sum([w.size for w in weights])
    print('Total number of weights: ', n_weights)

    # assert n_weights == n_inp * config.n_hidden + config.n_hidden * config.n_out, "n_weights is not properly calculated"

    if config.simulate_faults:
        percentage_stuck = config.percentage_stuck
        percentage_negative_change = config.percentage_negative_change
        percentage_positive_change = 1. - config.percentage_negative_change - config.percentage_stuck

        print(f'Faults-percentage. Stuck: {percentage_stuck}, pos. change: {percentage_positive_change}, '
              f'neg. change: {percentage_negative_change}')
        assert np.allclose(percentage_stuck + percentage_positive_change + percentage_negative_change, 1.), \
            "Percentages do not sum up to 1"

        fault_factor = generate_faults(n_weights, percentage_stuck, percentage_negative_change)
    else:
        fault_factor = np.ones(n_weights, dtype=float)

    # fault_factor is a single array variable with n_weights elements

    if config.save_data:
        save_train_dict = {'fault_factor': fault_factor}
        record_dict('train', save_train_dict)

    # Create mask variable, and artificial_weights
    # artificial_weights: for the pruning version where weights are frozen (and not removed by setting to 0)
    mask_weights = [np.ones_like(w) for w in weights]
    artificial_weights = [np.zeros_like(w) for w in weights]

    # Load the models of memristors and assign them randomly to each weight
    memristor_model = Memristor(config.w_max, config.delta_scaling_factor)
    memristor_dct = memristor_model.get_attributes_dict()
    if config.save_data:
        record_dict('memristor', memristor_dct)

    R = [set_R_to_get_w(w, memristor_model) for w in weights]

    # Prepare buffers
    history = [[] for _ in range(len(weights))]
    for cnt_layer in range(len(weights)):
        print(f'Layer shape: {weights[cnt_layer].shape}, layer size: {weights[cnt_layer].size}')
        for _ in range(weights[cnt_layer].size):
            history[cnt_layer].append([])
    history_noisy = copy.deepcopy(history)

    # Other params
    delta_scaling_factor = config.delta_scaling_factor
    stepwise_removal_percentage = 0.2
    if config.history == 0:  # history based on delta_w
        threshold = 0.03
    elif config.history == 1:  # history based on delta_R
        threshold = 0.1
    else:
        threshold = 0.

    train_it = -1
    last_iter_idx = int(math.ceil(1.0 * x_train.shape[0] / config.batch_size)) * config.epochs - 1
    for epoch in range(config.epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            train_it += 1
            # print('Num. samples, train_it: ', x_batch_train.shape[0], train_it)
            with tf.GradientTape() as tape:
                probs = model(x_batch_train, training=True)  # Probs for this minibatch
                loss_value = loss_fn(y_batch_train, probs)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric.update_state(y_batch_train, probs)
            curr_train_acc = train_acc_metric.result()
            train_acc_metric.reset_states()

            if train_it % config.accumulative_update_iter == 0 or train_it == last_iter_idx:
                # w --> R --> R_noisy --> w_noisy
                w_new = model.get_weights()[::2]  # Without noise

                cnt_weights_start = 0
                cnt_weights_end = 0
                for cnt_layer in range(len(w_new)):
                    cnt_weights_end += w_new[cnt_layer].size  # for indexing the fault_factor variable
                    w_new[cnt_layer], dR, noise, dR_noisy, R[cnt_layer] = update_memristive_vars(memristor_model,
                                                                                                 w_new[cnt_layer],
                                                                                                 mask_weights[cnt_layer],
                                                                                                 R[cnt_layer],
                                                                                                 delta_scaling_factor,
                                                                                                 fault_factor[
                                                                                                 cnt_weights_start:cnt_weights_end],
                                                                                                 (train_it > 0 and
                                                                                                  train_it % config.enforce_update_iter == 0)
                                                                                                 or train_it == last_iter_idx
                                                                                                 )
                    if config.to_prune:
                        idx_changed = np.where(dR != 0)
                        idx_flattened = np.ravel_multi_index(idx_changed, dR.shape)

                        # History buffers
                        if config.history == 0:  # history of delta_w
                            raise NotImplementedError
                        elif config.history == 1:  # history of delta_R
                            if idx_changed[0].size > 0:
                                lstdata = mask_weights[cnt_layer][idx_changed] * dR[idx_changed]
                                # Add element to history buffer
                                _ = list(map(lambda x, y: history[cnt_layer][x].append(y), idx_flattened, lstdata))
                                # Remove the "oldest" element from history buffer
                                _ = list(map(lambda x: history[cnt_layer][x].pop(0) if len(
                                    history[cnt_layer][x]) > config.history_window else history[cnt_layer][x],
                                             idx_flattened))

                                lstdata = mask_weights[cnt_layer][idx_changed] * dR_noisy[idx_changed]
                                # Add element to history_noisy buffer
                                _ = list(
                                    map(lambda x, y: history_noisy[cnt_layer][x].append(y), idx_flattened, lstdata))
                                # Remove the "oldest" element from history_noisy buffer
                                _ = list(map(lambda x: history_noisy[cnt_layer][x].pop(0) if len(
                                    history_noisy[cnt_layer][x]) > config.history_window else history_noisy[cnt_layer][
                                    x],
                                             idx_flattened))
                        else:
                            raise NotImplementedError

                        # Then perform test
                        if config.test_for_removing == 0:  # use use_welch_test
                            raise NotImplementedError
                        elif config.test_for_removing == 1:  # use paired t-test
                            raise NotImplementedError
                        else:  # use linear regression
                            if idx_changed[0].size > 0:
                                mask_weights[cnt_layer], artificial_weights[cnt_layer] = prune(history[cnt_layer],
                                                                 history_noisy[cnt_layer],
                                                                 mask_weights[cnt_layer],
                                                                 threshold,
                                                                 stepwise_removal_percentage,
                                                                 config.min_connectivity,
                                                                 fault_factor[cnt_weights_start:cnt_weights_end],
                                                                 idx_flattened,
                                                                 config.history_window,
                                                                 config.use_fixed_w,
                                                                 w_new[cnt_layer],
                                                                 artificial_weights[cnt_layer]
                                                                 )

                    cnt_weights_start = cnt_weights_end
                    # Copy _new to orig. vars
                    w_new[cnt_layer] *= mask_weights[cnt_layer]
                    w_new[cnt_layer] = np.clip(w_new[cnt_layer], -config.w_max, config.w_max)

                # Reset weights of the model
                combined_model_weights = model.get_weights()
                combined_model_weights[::2] = [a + b for a, b in zip(w_new, artificial_weights)]
                model.set_weights(combined_model_weights)

            # Log every x batches.
            if train_it % 100 == 0 or train_it == last_iter_idx:
                connectivity = np.sum(
                    np.array([np.sum(single_mask_mat) for single_mask_mat in mask_weights])) / n_weights

                print(f'Training loss (for one batch) at iter {train_it}: {float(loss_value):.4f}')
                print(f'Connectivity: {100 * connectivity}%')
                print(f'Iter {train_it}, accuracy: {float(curr_train_acc)}')

            if config.save_data:
                save_train_dict = {'loss': float(loss_value),
                                   'accuracy': float(curr_train_acc)
                                   }
                record_dict('train', save_train_dict)

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_probs = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_probs)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))

    print('Train it, last_iter_idx: ', train_it, last_iter_idx)
    assert train_it == last_iter_idx, "Last iter not calculated properly."

    # Run a test loop at the end of training
    for x_batch_test, y_batch_test in test_dataset:
        test_probs = model(x_batch_test, training=False)
        # Update test metrics
        test_acc_metric.update_state(y_batch_test, test_probs)
    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()
    print("Test acc: %.4f" % (float(test_acc),))

    if config.save_data:
        save_test_dict = {  # 'loss': float(loss_value),
            'accuracy': float(test_acc),
            'connectivity': connectivity,
        }
        record_dict('test', save_test_dict)

        save_weights_dict = {}
        for cnt_layer in range(len(mask_weights)):
            save_weights_dict['mask_w' + str(cnt_layer + 1)] = mask_weights[cnt_layer]
        record_dict('test', save_weights_dict)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--n-hidden", type=int, default=128, help="Number of hidden neurons to use")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (epoch = full dataset propagation")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr-cosine-decay", type=int, default=0, help="To use cosine lr decay (1) or not (0)?")
    parser.add_argument("--min-connectivity", type=float, default=0.1,
                        help="Min. connectivity in the network to keep per layer")
    parser.add_argument("--accumulative-update-iter", type=int, default=1,
                        help="Accumulate updates, update memristive vars every x iter")
    parser.add_argument("--history", type=int, default=1, help="History type. 0 for history based on dw; 1 for history based on dR")
    parser.add_argument("--history-window", type=int, default=10,
                        help="Number of last elements for the test (e.g. lin. reg.)")
    parser.add_argument("--to-prune", type=int, default=1, help="To prune (1) or not (0)?")
    parser.add_argument("--simulate-faults", type=int, default=1, help="To simulate faults (1) or not (0)?")
    parser.add_argument("--wmax", type=float, default=0.5, help="maximum range for weights, [-wmax, wmax]")
    parser.add_argument("--enforce-update-iter", type=int, default=100, help="When to enforce updates?")
    parser.add_argument("--use-fixed-w", type=int, default=0,
                        help="Keep the weights that should be removed fixed, i.e., frozen (1), or set them to zero (0)?")
    parser.add_argument("--using-laptop", type=int, default=1, help="Running on my laptop? (otherwise on office PC)."
                                                                    "When running on laptop, simulation data not saved.")

    args = parser.parse_args()

    to_save = 0
    if args.using_laptop == 0:
        to_save = 1

    # delta_lst = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    delta_lst = [0.01]

    # percentage_stuck_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 0.9]
    # percentage_negative_change_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 0.9]

    # When a particular scenario is of interest
    percentage_stuck_lst = [0.3]
    percentage_negative_change_lst = [0.3]

    for i, delta in enumerate(delta_lst):
        for perc_stuck in percentage_stuck_lst:
            for perc_neg in percentage_negative_change_lst:
                if perc_stuck + perc_neg > 0.9:
                    continue

                print('\n \n ------------------ Starting a new simulation ------------------ ')
                print(f'Delta: {delta}, perc_stuck: {perc_stuck}, perc_neg: {perc_neg}')
                print('\n \n')

                config = dict(
                    seed=args.seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=0.01,
                    lr_decay=0.99,
                    lr_decay_iter=1000,
                    lr_cosine_decay=args.lr_cosine_decay,
                    n_hidden=args.n_hidden,
                    n_out=10,
                    w_max=args.wmax,
                    min_connectivity=args.min_connectivity,
                    simulate_faults=args.simulate_faults,
                    accumulative_update_iter=args.accumulative_update_iter,
                    to_prune=args.to_prune,
                    test_for_removing=2,  # 0: Welch t-test (not impl.), 1: paired t-test (not impl.). 2: lin. reg.
                    delta_scaling_factor=delta,  # args.delta_scaling_factor
                    percentage_stuck=perc_stuck,
                    percentage_negative_change=perc_neg,
                    history_window=args.history_window,
                    history=args.history,
                    enforce_update_iter=args.enforce_update_iter,
                    use_fixed_w=args.use_fixed_w,
                    save_data=to_save
                )

                config = sdict(config)  # Also makes c immutable

                np.random.seed(config.seed)
                tf.random.set_seed(config.seed)

                datetime_suffix = datetime.now().strftime("D%d-%m-%Y-T%H-%M-%S")
                sim_name = "{}".format(datetime_suffix)
                name = 'ANN'
                if config.to_prune == 0:
                    subname = '/MNIST/to-prune-0'
                    # subname = '/MNIST/to-prune-0-analysis'

                else:
                    subname = '/MNIST/to-prune-1'
                    # subname = '/MNIST/to-prune-1-analysis'

                if args.using_laptop:
                    root_dir = os.path.expanduser(
                        os.path.join('/Users/ckraisnikovic/output/', 'Memristors/', name + subname))
                else:
                    root_dir = os.path.expanduser(os.path.join('/calc/ceca/output/', 'Memristors/', name + subname))
                root_dir_results = root_dir + '/' + sim_name + '/results'

                with SimManager(sim_name, root_dir, write_protect_dirs=False, tee_stdx_to='output.log') as simman:
                    paths = simman.paths
                    # Store config
                    with open(os.path.join(paths.data_path, 'config.yaml'), 'w') as f:
                        yaml.dump(config.todict(), f, allow_unicode=True, default_flow_style=False)

                    # Open recorder
                    datastore = ZarrDataStore(os.path.join(paths.results_path, 'data.mdb'))
                    recorder = Recorder(datastore)
                    print("Results will be stored in %s" % paths.results_path)

                    main(config)

                    # Close recorder
                    recorder.close()
                    print("Results stored in %s" % paths.results_path)

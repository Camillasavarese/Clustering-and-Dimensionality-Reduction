import numpy as np
import pickle
import random
import os

import utilities as u
import queries as q

N_TRACKS = 1413
HOP_SIZE = 512
OFFSET = 1.0
DURATION = 30  # TODO: to be tuned!
THRESHOLD = 0  # TODO: to be tuned!


def find_unique_peaks(peaks_position_total):
    ''' It takes a list of peaks values and return the unique values '''

    set_peaks = set()
    for i in range(len(peaks_position_total)):
        for j in range(len(peaks_position_total[i])):
            set_peaks.add(
                (peaks_position_total[i][j][0], peaks_position_total[i][j][1]))
    set_peaks = list(set_peaks)
    return set_peaks


def take_peaks():
    ''' Extracts peaks for each song and store, for each peak, its position in the track and its value '''

    track_list = q.ret_track_list()

    # To optimize our work we create this matrix only one time and then we store it so...
    # ...if it already exists we load it, else we build it rounding the peaks values to second decimal digit
    if "peaks_list" in os.listdir("/content/drive/My Drive/ADM-HW4/"):
        with open("/content/drive/My Drive/ADM-HW4/peaks_list", "rb") as f:
            peaks_position_total = pickle.load(f)
    else:
        peaks_position_total = []
        for el in track_list:
            peaks_position = []
            audio = "/content/drive/MyDrive/ADM-HW4/data/mp3_dataset/" + el + ".wav"
            track, sr, onset_env, peaks = u.load_audio_peaks(
                audio, OFFSET, DURATION, HOP_SIZE)
            for i in peaks:
                peaks_position.append((i, round(onset_env[i], 2)))
            peaks_position_total.append(peaks_position)

        # Write the matrix in a binary file
        with open("/content/drive/My Drive/ADM-HW4/peaks_list", "wb") as f:
            pickle.dump(peaks_position_total, f)

    set_p = find_unique_peaks(peaks_position_total)

    return peaks_position_total, set_p


def create_shingles_matrix(peaks_position_total, set_peaks):
    ''' Starting from the peaks we build the shingle matrix as follows:
        we have "songs indexes" on columns and peaks values on rows (actually, we use their indexes
        in the respective lists), then we place, for each song, 1 on the lines corresponding to the peaks that song has '''

    if "shingles" in os.listdir("/content/drive/My Drive/ADM-HW4/"):
        with open("/content/drive/My Drive/ADM-HW4/shingles", "rb") as f:  # Recupera la lista dal file
            shingles = pickle.load(f)
    else:
        shingles = np.zeros((len(set_peaks), len(peaks_position_total)))
        for i in range(len(peaks_position_total)):
            # It takes the unique peaks values for each song
            tmp_set = set([(el[0], el[1]) for el in peaks_position_total[i]])
            for val in tmp_set:
                shingles[set_peaks.index(val)][i] = 1

        with open("/content/drive/My Drive/ADM-HW4/shingles", "wb") as f:
            pickle.dump(shingles, f)

    return shingles


def create_signature_matrix(ns, shingles):
    ''' It shifts 30 times the matrix rows according to the numbers in random_perm and for each permutation
        takes, for each song (aka column), the number of the row in wich the first 1 value appears '''

    # If the permutations have already been generated it loads them, else it generates the permutations
    if "random_perm" in os.listdir("/content/drive/My Drive/ADM-HW4/"):
        with open("/content/drive/My Drive/ADM-HW4/random_perm", "rb") as f:  # Recupera la lista dal file
            random_perm = pickle.load(f)
    else:
        random_perm = [random.randint(1, len(shingles)-1) for _ in range(30)]
        # Scrive la lista su un file binario
        with open("/content/drive/My Drive/ADM-HW4/random_perm", "wb") as f:
            pickle.dump(random_perm, f)

    # As above
    if "signature" in os.listdir("/content/drive/My Drive/ADM-HW4/"):
        with open("/content/drive/My Drive/ADM-HW4/signature", "rb") as f:  # Recupera la lista dal file
            signature = pickle.load(f)
    else:
        signature = np.zeros((30, ns))
        tmp_shingles = shingles.copy()

        for i in range(len(random_perm)):
            for j in range(ns):
                # Takes the first occurrence of 1 and memorize its index
                signature[i][j] = int(np.where(tmp_shingles[:, j] == 1)[0][0])
            # Shift matrix rows
            tmp_shingles = np.roll(shingles, random_perm[i], axis=0)

        # Write the signature matrix in a binary file
        with open("/content/drive/My Drive/ADM-HW4/signature", "wb") as f:
            pickle.dump(signature, f)

    return signature


def create_buckets(signature, ns, b=10):
    ''' It creates buckets considering 3 bands of length 10 '''

    buckets = dict()
    for i in range(0, len(signature), b):
        # Matrix with 10 rows (size of the band) and 1413 columns (songs)
        tmp_signature = signature[i:i+b]
        for j in range(ns):
            # Signature (piece of length 10) of the j-th song
            tmp = tuple(map(int, tmp_signature[:, j]))
            # Entire signature (of length 10) of the j-th song
            tmp_sign = list(map(int, signature[:, j]))
            res = u.find_best(tmp, buckets)
            if res != 0:
                buckets[res].append((j, tmp_sign))
            # If no bucket have a similarity greater than 0.5 (our treshold) we create a new one
            else:
                buckets[tmp] = [(j, tmp_sign)]

    # Write buckets in a binary file
    with open("/content/drive/My Drive/ADM-HW4/buckets", "wb") as f:
        pickle.dump(buckets, f)

    return buckets

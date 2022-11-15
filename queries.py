import numpy as np
import pickle
import os

import utilities as u

N_TRACKS = 1413
HOP_SIZE = 512
OFFSET = 1.0
DURATION = 30  # TODO: to be tuned!
THRESHOLD = 0  # TODO: to be tuned!


def take_peaks():
    ''' Extracts peaks for each query song and store, for each peak, its position in the track and its value'''

    peaks_position_query = []
    p = "/content/drive/MyDrive/ADM-HW4/data/tracks/"
    for i in range(1, len(os.listdir(p))+1):
        peaks_position = []
        audio = p + "track" + str(i) + ".wav"
        track, sr, onset_env, peaks = u.load_audio_peaks(
            audio, OFFSET, DURATION, HOP_SIZE)
        for i in peaks:
            peaks_position.append((i, round(onset_env[i], 2)))
        peaks_position_query.append(peaks_position)

    return peaks_position_query


def create_shingles(set_peaks, peaks_position_query):
    ''' Create shingle matrix for queries as we did for other songs '''

    shingles_query = np.zeros((len(set_peaks), len(peaks_position_query)))
    for i in range(len(peaks_position_query)):
        tmp_set = set([(el[0], el[1]) for el in peaks_position_query[i]])
        for val in tmp_set:
            # If this peak is not in those calculated for the songs of our dataset, ignore that
            try:
                shingles_query[set_peaks.index(val)][i] = 1
            except:
                pass

    with open("/content/drive/My Drive/ADM-HW4/shingles_query", "wb") as f:
        pickle.dump(shingles_query, f)

    return shingles_query


def create_signature(peaks_position_query, shingles_query):
    ''' It shifts 30 times the matrix rows according to the numbers (the same as before) in random_perm and for each permutation
        takes, for each song (aka column), the number of the row in wich the first 1 value appears '''

    with open("/content/drive/My Drive/ADM-HW4/random_perm", "rb") as f:
        random_perm = pickle.load(f)

    signature_query = np.zeros((len(random_perm), len(peaks_position_query)))
    tmp_shingles = shingles_query.copy()
    for i in range(len(random_perm)):
        for j in range(len(peaks_position_query)):
            # Takes the first occurrence of 1 and memorize its index
            signature_query[i][j] = int(
                np.where(tmp_shingles[:, j] == 1)[0][0])
        # Shift matrix rows
        tmp_shingles = np.roll(shingles_query, random_perm[i], axis=0)
    # np.set_printoptions(suppress = True)
    return signature_query


def insert_in_buckets(signature_query, peaks_position_query, b=10):
    ''' Find the correct bucket in which to search the query song '''

    with open("/content/drive/My Drive/ADM-HW4/buckets", "rb") as f:
        buckets = pickle.load(f)

    results = set()
    for i in range(0, len(signature_query), b):
        # Matrix with 10 rows (size of the band) and 1413 columns (songs)
        tmp_signature = signature_query[i:i+b]
        for j in range(len(peaks_position_query)):
            tmp = tuple(map(int, tmp_signature[:, j]))
            for bu in buckets:
                if u.normalize(tmp, bu) > 0.5:
                    t = buckets[bu][0][0]
                    results.add((j, t))

    return results


def ret_track_list():
    ''' Return the tracks list '''

    with open("/content/drive/MyDrive/ADM-HW4/data/mp3_dataset/all.list", 'r') as f:
        track_list = f.read().split('\n')
        track_list = track_list[:len(track_list)-1]
    return track_list


def print_res(results):
    ''' Print results of the LSH '''

    track_list = ret_track_list()
    results = sorted(list(results), key=lambda x: x[0])
    for el in results:
        print(el[0]+1, track_list[el[1]])


def alternative_LSH(nq, nt, signature_query, signature, THRESHOLD=0.5):
    ''' Our alternative version to LSH '''

    track_list = ret_track_list()
    for i in range(nq):
        track_query = signature_query[:, i]
        for j in range(nt):
            if u.normalize(track_query, signature[:, j]) > THRESHOLD:
                print(i+1, track_list[j])

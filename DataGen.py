import os
import h5py
import matplotlib.pyplot as plt
import os.path
import pypianoroll as piano
import numpy as np

def divide_into_bars(track,resolution,length,values):
    if track.size:
        bars = np.vsplit(track,track.shape[0]/resolution)
    else:
        empty_track = np.zeros((length,84))
        bars = np.vsplit(empty_track,empty_track.shape[0]/resolution)
    return np.asarray(divide_into_phrase((bars)))
def divide_into_phrase(bars):
    phrases = []
    phrase=[]
    n = 0
    for i in range(len(bars)):
        phrase.append(bars[i])
        n+=1
        if(n%4==0):
            phrases.append(np.array(phrase))
            phrase = []
            n = 0
    phrases = np.array(phrases)
    return phrases



def load_data(folder):
    data = []
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in [f for f in filenames if f.endswith(".npz")]:
            pianoroll = piano.Multitrack(os.path.join(dirpath, filename))
            duration = max(roll.pianoroll.shape[0] for roll in pianoroll.tracks)
            values =  max(roll.pianoroll.shape[1] for roll in pianoroll.tracks)
            multitrack_bar = []
            for track in sorted(pianoroll.tracks, key=lambda x: x.name):
                #print(track.pianoroll.shape)
                phrases = divide_into_bars(track.pianoroll[:,0:84],pianoroll.beat_resolution,duration,values)
                multitrack_bar.append(phrases)
            multitrack_bar = np.asarray(multitrack_bar)
            multitrack_bar = multitrack_bar.transpose((1,0,2,3,4))
            data.append(multitrack_bar)
    data = np.vstack(data)
    return data
# rolls = load_data('test2')
# fig = piano.plot(rolls[1])
# plt.show()
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'














#
# np.savez_compressed('dataset',data)

import os
import pypianoroll as ppr
import numpy as np
import matplotlib.pyplot as plt




def numpy_to_pianoroll(folder):
    #Bass Drums Guitar Piano Strings
    programs = [34,0,30,1,51]
    names = ['Bass' ,'Drums' ,'Guitar' ,'Piano' ,'Strings']
    tempo = np.full((96), 105)
    for filename in os.listdir(folder):
        multisample = np.load(os.path.join(folder,filename))

        for sample,i  in zip(multisample,range(multisample.shape[0])):
            tracks = []
            for instrument,program,name in zip(sample,programs,names):
                print(instrument.shape)
                track = np.vstack(instrument)
                print(track.shape)
                track[track > 0.5] = 100
                track[track < 0.5] = 0
                print(track.shape)
                track = np.pad(track.astype(int),((0,0),(0,44)),mode='constant')
                print(ppr.metrics.qualified_note_rate((track),100))
                print(ppr.metrics.n_pitches_used((track)))
                print(track.shape)
                isdrum = False
                if program == 0:
                    isdrum = True
                ppr_track = ppr.Track(track,program,isdrum,name)
                tracks.append(ppr_track)
            ppr_song = ppr.Multitrack(tracks=tracks, tempo=tempo, beat_resolution=24)

            print(123)
            plot = ppr.plot_multitrack(ppr_song,mode='separate',ytick='off')
            plt.savefig('gen_samples/'+filename+str(i)+".png",dpi=400)
            ppr.write(ppr_song, 'gen_samples/'+filename+"song")



#data= load_data("test")
numpy_to_pianoroll("generated")
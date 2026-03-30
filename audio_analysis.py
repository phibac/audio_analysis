# Import
from scipy.io import wavfile
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# read in the wav file and get the sampling rate
def audioanalysis():
    print("Audio analysis started...")

    # file is in root folder directory
    openDirectory = os.getcwd()

    # file is in another directory
    # openDirectory = r""

    for path, subdirs, files in os.walk(openDirectory):
        print('PATH: ', path)
        print('SUBDIRS: ', subdirs)
        print('FILES: ', files)

        for filenames in files:
            # if 'audio.wav' in os.path.join(path, subdirs, filenames):
            if 'recording.wav' in  filenames:

                audiofile_name = filenames

                samplerate, audiofile = wavfile.read(audiofile_name) # Get samplerate from audio device and audio data

                # Audio Data
                audio_original = []
                audio_data = []

                audio_original = np.array(audiofile)
                len_audio_original = len(audio_original)

                # Smoothing the audio signal
                window = 10
                weights = np.repeat(1.0, window)/window
                audio_original = np.convolve(audio_original, weights, mode='same')

                # Create time points for original audio plot
                videoFile_length = len(audio_original)
                video_length = videoFile_length/samplerate
                # video_length_rounded = round(videoFile_length/48000)
                duration = video_length
                time = np.linspace(0,duration,len(audio_original))

                # Eliminate noice in original audio signal
                for i in range (0, len(audio_original)):
                    if -50 < audio_original[i] < 50:
                        audio_original[i] = 0

                # Find repetitions for fft window with specific size
                window = 100
                intcheck = False
                while not intcheck:
                    result = int(len(audio_original)/window)
                    intcheck = isinstance(result,int)
                    if intcheck == True:
                        # print(result)
                        break

                fft_max_values = []
                freq_of_max_values = []

                # Start FFT for every window
                for i in range(0,result):
                    fourierTransform = np.fft.fft(audio_original[i*window:(i+1)*window])/len(audio_original[i*window:(i+1)*window])
                    fourierTransform = fourierTransform[range(int(len(audio_original[i*window:(i+1)*window])/2))]
                    tpCount     = len(audio_original[i*window:(i+1)*window])
                    values      = np.arange(int(tpCount/2))
                    timePeriod  = tpCount/samplerate
                    frequencies = values/timePeriod

                    fft_values = abs(fourierTransform)
                    
                    fft_max_value = max(abs(fourierTransform)) # Find max amplitude in window
                    fft_max_values.append(fft_max_value) # Add max amplitude to list

                    freq_max = np.array(np.where(fft_values == fft_max_value)) # Find f of max amplitude in window

                    freq_of_max_values.append(frequencies[freq_max[0,0]]) 

                all_freq = freq_of_max_values

                ##############################################################################

                # Define Range for frequencies -> add here tolerance if needed
                for i in range(0,len(freq_of_max_values)):
                    if freq_of_max_values[i] > 2400:
                        freq_of_max_values[i] = 0
                    if freq_of_max_values[i] < 1920:
                        freq_of_max_values[i] = 0

                # Set single high to zero
                for i in range(0,len(freq_of_max_values)):
                    if i-1 > 0 and i+1 < len(freq_of_max_values):
                        if freq_of_max_values[i-1] == 0 and freq_of_max_values[i] != 0 and freq_of_max_values[i+1] == 0:
                            freq_of_max_values[i] = 0

                # Split in two frequencies

                ix1 = [] # index where f = 2400
                val1 = [] # value 

                for i in range(0,len(freq_of_max_values)):
                    if freq_of_max_values[i] == 2400:
                        ix1.append(i)
                        val1.append(2400) ####### 1000

                y1 = [0]*len(freq_of_max_values)
                for i in range(0,len(ix1)):
                    y1[ix1[i]] = 2400

                # Set single high to zero
                for i in range(0,len(y1)):
                    if i-1 > 0 and i+1 < len(y1):
                        if y1[i-1] == 0 and y1[i] != 0 and y1[i+1] == 0:
                            y1[i] = 0

                for i in range(0,len(y1)):
                    if i-1 > 0 and i+100 < len(y1):
                        if y1[i-1] == 0 and y1[i] != 0 and y1[i+1] == 0:
                            y1[i] = 0

                # Window
                window_y1 = 10
                intcheck = False
                while not intcheck:
                    result = int(len(y1)/window_y1)
                    intcheck = isinstance(result,int)
                    if intcheck == True:
                        print(result)
                        break

                for i in range(0,result):
                    zero = 0
                    value = 0

                    for k in range(0,window_y1):
                        diff = y1[(i*window_y1)+k]
                        if diff == 0:
                            zero +=1
                        else:
                            value += 1
                            val = diff
                            
                    if zero > value:
                        for k in range(0,window_y1):
                            y1[(i*window_y1)+k] = 0

                    if zero < value:
                        for k in range(0,window_y1):
                            y1[(i*window_y1)+k] = val

                    if zero == value:
                        for k in range(0,window_y1):
                            y1[(i*window_y1)+k] = val

                # Split in two frequencies

                ix2 = []
                val2 = []

                for i in range(0,len(freq_of_max_values)):
                    if freq_of_max_values[i] == 1920:
                        ix2.append(i)
                        val2.append(1920) ####### 1000

                y2 = [0]*len(freq_of_max_values)
                for i in range(0,len(ix2)):
                    y2[ix2[i]] = 1920

                # Set single high to zero
                for i in range(0,len(y2)):
                    if i-1 > 0 and i+1 < len(y2):
                        if y2[i-1] == 0 and y2[i] != 0 and y2[i+1] == 0:
                            y2[i] = 0

                for i in range(0,len(y2)):
                    if i-1 > 0 and i+100 < len(y2):
                        if y2[i-1] == 0 and y2[i] != 0 and y2[i+1] == 0:
                            y2[i] = 0

                # Window
                window_y2 = 10
                intcheck = False
                while not intcheck:
                    result = int(len(y2)/window_y2)
                    intcheck = isinstance(result,int)
                    if intcheck == True:
                        print(result)
                        break

                for i in range(0,result):
                    zero = 0
                    value = 0

                    for k in range(0,window_y2):
                        diff = y2[(i*window_y2)+k]
                        if diff == 0:
                            zero +=1
                        else:
                            value += 1
                            val = diff
                            
                    if zero > value:
                        for k in range(0,window_y2):
                            y2[(i*window_y2)+k] = 0

                    if zero < value:
                        for k in range(0,window_y2):
                            y2[(i*window_y2)+k] = val

                    if zero == value:
                        for k in range(0,window_y2):
                            y2[(i*window_y2)+k] = val

                # Count how often up and down
                up = 0
                down = 0

                # Index of up and down
                t_up = []
                t_down = []

                for i in range(0,len(y1)):
                    if i+1 < len(y1):
                        if y1[i] == 0 and y1[i+1] != 0:
                            up +=1 # Count Up
                            t_up.append(i) # Add index of up
                        if y1[i] != 0 and y1[i+1] == 0:
                            down +=1 # Count Down
                            t_down.append(i) # Add index of down

                for i in range(0,up):
                    t_up[i] = ((t_up[i]/samplerate)*window)

                for i in range(0,down):
                    t_down[i] = (t_down[i]/samplerate)*window

                result_up_down = []
                for i in range(0,up):
                    result_up_down.append(abs(t_up[i] - t_down[i]))

                if up == 4:
                    print("Starting position detected: " + str(result_up_down[0]) + 'seconds ' + str(result_up_down[1]) + 'seconds ' + str(result_up_down[2]) + 'seconds ' )
                    print("End detected: " + str(result_up_down[3]) + 'seconds')
                else:
                    print('Please check audio data.')

                t_up2 = 0
                t_down2 = 0

                up2 = []
                down2 = []

                for i in range(0,len(y2)):
                    if i+1 < len(y2):
                        if y2[i] == 0 and y2[i+1] != 0:
                            t_up2 +=1
                            up2.append(i)
                        if y2[i] != 0 and y2[i+1] == 0:
                            t_down2 +=1
                            down2.append(i)

                for i in range(0,t_up2):
                    up2[i] = ((up2[i]/samplerate)*window)

                for i in range(0,t_down2):
                    down2[i] = (down2[i]/samplerate)*window

                result_up_down2 = []

                for i in range(0,t_up2):
                    result_up_down2.append(abs(up2[i] - down2[i]))

                if len(up2) == 1 and len(down2) == 1:
                    print("Exposure detected: " + str(result_up_down2[0]) + ' seconds')
                    error = 0
                else:
                    error = 1
                    print('Please check audio data.')

                ## Plot Results
                print(error)
                if error == 0:
                    ## Plot Audio Analysis Result
                    plt.figure(1)
                    plt.subplot(3,1,1)
                    time1 = np.linspace(0,duration,len(audio_original))
                    plt.plot(time1,audio_original)
                    time2 = np.linspace(0,duration,len(y1))
                    plt.plot(time2,y1,color='orange')
                    plt.legend(['Unprocessed Audio data'],loc='upper left')
                    plt.title("Detection of sequence with frequence 1")


                    plt.subplot(3,1,2)
                    time1 = np.linspace(0,duration,len(audio_original))
                    plt.plot(time1,audio_original)
                    time2 = np.linspace(0,duration,len(y1))
                    plt.plot(time2,y2,color='orange')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Amplitude')
                    plt.legend(['Unprocessed Audio Data'],loc='upper left')
                    plt.title("Detection of sequence with frequence 2")
                    
                    plt.subplot(3,1,3)
                    time2 = np.linspace(0,duration,len(y1))
                    plt.plot(time2,y1,color='orange')
                    plt.plot(time2,y2,color='orange')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Amplitude')
                    plt.title('Merge detected sequences with frequence 1 + 2')
                    plt.tight_layout()
                    plt.show()

                else:
                    plt.figure(1)
                    time1 = np.linspace(0,duration,len(audio_original))
                    plt.plot(time1,audio_original)
                    plt.show()

if __name__=='__main__':
    audioanalysis()
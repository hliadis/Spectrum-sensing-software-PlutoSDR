import numpy as np
import adi
import matplotlib.pyplot as plt
import math
import sys
import time
import datetime

CH_48_BW = 80.0e6 # Hz
FM_BW = 120e3 # Hz
LORA_BW = 125e3 # Hz

# This function computes the fft of the given rx_buffer. 
# Right after that it computes the Power Spectral Density (PSD)
# in dB through the fft sequence
def psd(rx_samples):
    fft_rx = np.fft.fft(rx_samples)
    scale = 2.0/(len(rx_samples) * len(rx_samples))
    psd = scale * (fft_rx.real**2 + fft_rx.imag**2)
    psd_log = 10.0*np.log10(psd)
    psd_shifted = np.fft.fftshift(psd_log)
    return(psd_shifted)

# This function returns the index of the element in array that contains
# the value which is closer to the given number
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return(idx)

def lora(sdr):
    sample_rate = 1e6 # Hz
    center_freq = 866.1e6 # Hz
    num_samps = 100000 # number of samples per call to rx()

    sdr.sample_rate = int(sample_rate)

    # Config Rx
    sdr.rx_lo = int(center_freq)
    sdr.rx_rf_bandwidth = int(20e6)
    sdr.rx_buffer_size = num_samps
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 64.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC

    # Calculate noise floor by tuning SDR in a frequency with no transmission
    sdr.rx_lo = int(780e6)
    rx_samples = sdr.rx()
    
    noise_floor = np.mean( psd(rx_samples)*np.blackman(len(psd(rx_samples))) )
    print("**************************")
    print("Noise floor: ", noise_floor)
    print("**************************")
    print("\n")
    sdr.rx_lo = int(center_freq)

    start_time = time.time()
    while time.time() - start_time < 0.5:
        rx_samples = sdr.rx()

    psd_shifted = psd(rx_samples*np.blackman(len(rx_samples)))
        
    fft_fr = np.fft.fftshift( np.fft.fftfreq(len(rx_samples), d=1/sample_rate) )

    transmission_freq = 0
    start = stop = 0
    
    start = find_nearest(fft_fr, value = (-LORA_BW/2))
    stop = find_nearest(fft_fr, value = (+LORA_BW/2))
    
    transmission_freq = math.ceil(((fft_fr[stop] - fft_fr[start]) / 2) + fft_fr[start]) + sdr.rx_lo
    transmission_freq = round(transmission_freq / 1e6 , 1)
                                
    avg = np.mean(psd_shifted[start:stop])

    snr_dB = avg - noise_floor

    if(snr_dB > 15):
        print("transmission frequency: " + str(transmission_freq) + "MHz")
        print("Average: " + str(round(avg, 2)) + " dB")
        print("\n")
        
    plt.figure(1)
    plt.plot(fft_fr/1e6, psd_shifted)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD")
    plt.show()


def fm_band(sdr, left_limit, right_limit):
    sample_rate = 1e6 # Hz
    center_freq = 87.9e6 # Hz
    num_samps = 100000 # number of samples per call to rx()
    
    # Initialize LP filter dictionary and lists for various
    # time measurements
    dict = {}
    single_transm_time = []
    whole_scan_time = []
    rx_lo_change_time = []
    rx_lo_change_freq = []
    sampling_time = []
    frequency_table = []
    fm_signals = []
    trans_freq_table = []
    counter = 0

    # Config Rx
    sdr.sample_rate = int(sample_rate)
    sdr.rx_lo = int(center_freq)
    sdr.rx_rf_bandwidth = int(20e6)
    sdr.rx_buffer_size = num_samps
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 64.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC

    # Calculate noise_floor by tuning SDR in a frequency with no transmission
    sdr.rx_lo = int(82e6)

    rx_samples = sdr.rx()
    rx_samples = rx_samples*np.blackman(len(rx_samples))
    noise_table = psd(rx_samples)
    noise_floor = np.mean(noise_table)
    print("**************************")
    print("Noise floor: ", noise_floor)
    print("**************************")
    print("\n")
    
    # Number of center frequencies (equal spaced) within 1MHz
    num = int((sample_rate/100e3) + 1)
    
    #Create an equal spaced array of frequencies within len(sample_rate)
    f = np.linspace(-sdr.sample_rate/2, sdr.sample_rate/2, num)
    
    # Get array of frequency values of each FFT bin
    fft_fr = np.fft.fftfreq(len(rx_samples), d=1/sample_rate)
    fft_fr = np.fft.fftshift(fft_fr)

    # Fill dictionary with 70kHz low_pass filters for each frequency within 1MHz analysis
    for i in range(num):
        if (i == 0):
            stop = find_nearest(fft_fr, value = (f[i] + FM_BW/2))
            dict[i] = list()
            dict[i].append(stop)
        
        elif (i == num - 1):
            start = find_nearest(fft_fr, value = (f[i] - FM_BW/2)) 
            dict[0].insert(0,start)
        
        else:
            start = find_nearest(fft_fr, value = (f[i] - FM_BW/2))
            stop = find_nearest(fft_fr, value = (f[i] + FM_BW/2))
            dict[i] = list()
            dict[i].append(start)
            dict[i].append(stop)

    sdr.rx_lo = int(center_freq)

    reg_end = 0
    transmission_freq = 0
    reg_time = 0

    while True:
        
        rx_lo_change_freq.append(87.9 + counter)
        rcv_time = time.time()
        
        # Receive I, Q data for 500msec
        start = datetime.datetime.now()
        while time.time() - rcv_time < 0.5:
            rx_samples = sdr.rx()
        
        # Apply Blackman window in order to avoid sudden transitions
        # between the first and the last sample 
        rx_samples = rx_samples*np.blackman(len(rx_samples))
        
        # Calculate fft and psd
        psd_shifted = psd(rx_samples)
        sampling_time.append( (datetime.datetime.now() - start).total_seconds()*1000 )

        store_time = list()

        # Conduct 70kHz analysis for every frequency within 1MHz bandwidth
        for i in range(num):
            
            #left limit of frequency array
            if(i == 0):

                if(reg_end == 0):
                    continue

                start_1 = datetime.datetime.now()

                # Compute the psd mean value of samples within [0:35e3] kHz.
                # Add the psd mean value of the remaining samples stored in the 
                # register reg_end in order to complete the analysis for this frequency point.
                stop = dict[i][1]
                avg = np.mean(psd_shifted[0:stop]) + reg_end
                
                start_1 = datetime.datetime.now() - start_1

                single_transm_time.append(start_1.total_seconds()*1000 + reg_time)
                store_time.append(single_transm_time[-1])
                
                # Get center frequency of the fft bins within the 70kHz bandwidth
                transmission_freq = math.ceil((fft_fr[0]) + sdr.rx_lo)
                transmission_freq = round(transmission_freq / 1e6 , 1)
                frequency_table.append(transmission_freq)
            
            #right limit of frequency array
            elif(i == num - 1):
                reg_time = datetime.datetime.now()

                start = dict[0][0]

                # Compute the psd mean value for the elements psd_shifted[(num_samps - 35e3):num_samps]
                # and store the mean value to the register reg_end. In the next 1MHz interval, the register
                # will be added with the psd mean value of the fft bins within [0:35e3] Hz starting from the 
                # left limit of the new interval
                reg_end = np.mean( psd_shifted[start:num_samps] )

                reg_time = (datetime.datetime.now() - reg_time).total_seconds()*1000
                continue
            
            # if 0 < i < num conduct 70kHz scan for this frequency
            else:
                
                start_1 = datetime.datetime.now()
                
                start = dict[i][0]
                stop = dict[i][1]

                avg = np.mean(psd_shifted[start:stop])
                
                single_transm_time.append( (datetime.datetime.now() - start_1).total_seconds()*1000 )
                store_time.append(single_transm_time[-1])
                
                # Get center frequency of the fft bins within the 70kHz bandwidth
                transmission_freq = math.ceil(((fft_fr[stop] - fft_fr[start]) / 2) + fft_fr[start]) + sdr.rx_lo
                transmission_freq = round(transmission_freq / 1e6 , 1)
                frequency_table.append(transmission_freq)
            
            # Analyze only frequencies which are equal or greater than the given left limit of the total interval
            if( transmission_freq < float(left_limit) ):
                continue

            # Right limit of total interval reached. plot time tables before ending the execution
            elif( transmission_freq > float(right_limit) ):
                
                print("----------TIME_METRICS----------")
                print("sampling_time: " +  str(round( np.mean(sampling_time) , 3) ) + " msec")
                print("whole_scan_time: " +  str(round( np.mean(whole_scan_time) , 3) ) + " msec")
                print("single_scan_time: " + str(round( np.mean(single_transm_time), 3) ) + " msec")
                print("rx_lo_change_time: " +  str(round( np.mean(rx_lo_change_time) , 3) ) + " msec")
                print("-------------------------------\n")
                
                #--------------70kHz_PLOT------------------------
                plt.figure(1)
                plt.plot(frequency_table,single_transm_time, marker='o')
                plt.axhline(y=round( np.mean(single_transm_time) , 3), color='r', linestyle='-',label='mean_value')
                l = plt.legend(loc ='upper right')
                plt.xlabel("Frequency [MHz]")
                plt.ylabel("Time [msec]")

                for i in range(len(fm_signals)):

                    plt.annotate(str(trans_freq_table[i]),
                                xy=(trans_freq_table[i], single_transm_time[ frequency_table.index(trans_freq_table[i]) ]))
                plt.title("70kHz scan time for every frequency")
                plt.show()
                
                #--------------Transmission_freq_PLOT-------------
                plt.figure(1)
                plt.plot(trans_freq_table,fm_signals,marker='o')
                plt.xlabel("Transmission frequencies [MHz]")
                plt.ylabel("PSD")

                for i in range(len(fm_signals)):

                    plt.annotate(str(trans_freq_table[i]),
                                xy=(trans_freq_table[i], fm_signals[i]))
                plt.axhline(y=(noise_floor + 15), color='r', linestyle='-',label='SNR threshold')
                l = plt.legend(loc ='upper right')
                plt.title("FM stations")
                plt.show()

                #--------------rx_lo_change_PLOT------------------
                plt.figure(1)
                plt.plot(rx_lo_change_freq[:len(rx_lo_change_freq)-1], rx_lo_change_time, marker='o')
                plt.axhline(y=round( np.mean(rx_lo_change_time) , 3), color='r', linestyle='-',label='mean_value')
                l = plt.legend(loc ='upper right')
                plt.xlabel("Frequency [MHz]")
                plt.ylabel("Time [msec]")
                for i in range(len(rx_lo_change_freq)-1):

                    plt.annotate(str(rx_lo_change_freq[i]),
                                xy=(rx_lo_change_freq[i], rx_lo_change_time[i]))
                plt.title("rx_lo change time")
                plt.show()

                #--------------sampling_time_PLOT------------------
                plt.figure(1)
                plt.plot(rx_lo_change_freq,sampling_time, marker='o')
                plt.axhline(y=round( np.mean(sampling_time) , 3), color='r', linestyle='-',label='mean_value')
                l = plt.legend(loc ='upper right')
                plt.xlabel("Frequency [MHz]")
                plt.ylabel("Time [msec]")
                for i in range(len(rx_lo_change_freq)):

                    plt.annotate(str(rx_lo_change_freq[i]),
                                xy=(rx_lo_change_freq[i], sampling_time[i]))
                plt.title("Sampling & PSD computation time")
                plt.show()

                #-----------whole_scan_PLOT-------------------------
                plt.figure(1)
                plt.plot(rx_lo_change_freq[:len(rx_lo_change_freq)-1],whole_scan_time, marker='o')
                plt.axhline(y=round( np.mean(whole_scan_time) , 3), color='r', linestyle='-', label='mean_value')
                l = plt.legend(loc ='upper right')
                plt.xlabel("Frequency [MHz]")
                plt.ylabel("Time [msec]")
                for i in range(len(rx_lo_change_freq)-1):

                    plt.annotate(str(rx_lo_change_freq[i]),
                                xy=(rx_lo_change_freq[i], whole_scan_time[i]))
                plt.title("1MHz scan time")
                plt.show()

                sys.exit()

            # Compute Signal to Noise Ratio (SNR) for a given signal. It
            # is expressed in decibels
            snr_dB = avg - noise_floor

            # If SNR is greater than the given threshold, we decide that the level of 
            # our signal in this frequency is high enough to be considered as a transmission
            # signal.
            if(snr_dB > 15):
                
                trans_freq_table.append(transmission_freq)
                fm_signals.append(round(avg, 2))
                print("transmission frequency: " + str(transmission_freq) + "MHz")
                print("snr_dB: ", snr_dB)
                print("Average: " + str(round(avg, 2)) + " dB")
                print("\n")

        whole_scan_time.append(round(np.sum(store_time), 3))

        # tune SDR to the next center frequency, which is centered to the 1MHz interval.
        counter += int(sample_rate/1e6)
        next = (87.9 + counter)*1e6

        start = datetime.datetime.now()
        sdr.rx_lo = int(next)
        rx_lo_change_time.append( (datetime.datetime.now() - start).total_seconds()*1000 )


def wifi_band(sdr):

    # This dictionary contains all the wi-fi channels that participated in wi-fi experiments and analysis.
    wifi_channels = {"channel_1" : 2.412e9 ,
                     "channel_2" : 2.417e9 ,
                     "channel_3" : 2.422e9 ,
                     "channel_4" : 2.427e9 ,
                     "channel_5" : 2.432e9 ,
                     "channel_6" : 2.437e9 ,
                     "channel_7" : 2.442e9 ,
                     "channel_8" : 2.447e9 ,
                     "channel_9" : 2.452e9 ,
                     "channel_10" : 2.457e9 ,
                     "channel_11" : 2.462e9 ,
                     "channel_12" : 2.467e9 ,
                     "channel_13" : 2.472e9 ,
                     "channel_42" : 5.210e9 ,
                     "channel_48" : 5.240e9 ,
                     "channel_106": 5.530e9 ,
                     "channel_46" : 5.220e9 ,
                     "no_signal_5g" : 5.400e9}
    
    sample_rate = 10e6 # Hz
    num_samps = 100000 # number of samples per call to rx()
    sdr.sample_rate = int(sample_rate)

    # Config Rx
    sdr.rx_rf_bandwidth = int(20e6)
    sdr.rx_buffer_size = num_samps
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 50.0  # dB, increase to increase the receive gain, but be careful not to saturate the ADC

    sdr.rx_lo = int(5.400e9)
    
    # Calculate noise_floor by tuning SDR in a frequency with no transmission
    rx_samples = sdr.rx()

    noise_floor = np.mean(psd(rx_samples))
    print("**************************")
    print("Noise floor: ", noise_floor)
    print("**************************")
    print("\n")

    # Get array of frequency values of each FFT bin
    fft_fr = np.fft.fftshift( np.fft.fftfreq(len(rx_samples), d=1/sample_rate) )
    
    # Initialize values and lists for time measurements
    total_psd = 0
    time_axis = []
    avg_psd_axis = []
    sampling_time = []
    total_exec_time = []
    latency_time = 0

    # Iterate dictionary
    for key, value in wifi_channels.items():
        
        # If key is equal to the desired channel, start analysis
        if(key == "channel_42"):
            shift_value = 0
            temp = value + (-CH_48_BW / 2) + (sample_rate / 2)
            
            start = datetime.datetime.now()
            store_time = list()

            #Conduct 80MHz analysis for n seconds
            while (datetime.datetime.now() - start).total_seconds() < 60:
                total_psd = 0
                shift_value = 0
                start_exec = time.time()
                
                # Compute average psd of all the samples within
                # sample_rate range. Repeat until the whole wi-fi
                # channel bandwidth is analyzed 
                while CH_48_BW - shift_value > 0:
                
                    center_freq =  temp + shift_value
                    sdr.rx_lo = int(center_freq)

                    start_time = time.time()
                    
                    if(latency_time):
                        store_time.append(time.time() - latency_time)
                    
                    sample_time = time.time()
                    
                    rx_samples = sdr.rx()
                    
                    latency_time = time.time()

                    sample_time = time.time() - sample_time
                    sampling_time.append(sample_time)
                    
                    total_psd += np.mean(psd(rx_samples))

                    shift_value += sample_rate


                time_axis.append( (datetime.datetime.now() - start).total_seconds() )
                avg_psd_axis.append(total_psd / (CH_48_BW / sample_rate))
                total_exec_time.append(time.time() - start_exec)

                # After channel analysis, print some critical time values for evaluation
                print("channel: " + str(key) + " signal_power: " + str(total_psd / (CH_48_BW / sample_rate)) +'\n')
                print("simple latency: ", store_time[0])
                print("latency: ", np.sum(store_time[0:int((CH_48_BW / sample_rate))]))
                print("Average sampling time: ", np.mean(sampling_time))
                print("Total execution time: ", np.mean(total_exec_time))

            # Plot channel behaviour for elapsed time of n seconds
            plt.figure(1)
            plt.plot(time_axis, avg_psd_axis)
            plt.xlabel("Elapsed_time sec")
            plt.ylabel("PSD")
            plt.show()



if __name__ == '__main__':

    sdr = adi.Pluto("ip:192.168.3.1")

    if(sys.argv[1] == "wifi"):
        wifi_band(sdr)
    
    elif(sys.argv[1] == "lora"):
        lora(sdr)

    elif(sys.argv[1] == "fm"):

        try:
            print("\n*************************************")
            print("Enter the desired fm band interval,")
            print("input format: X.Y (frequency value in MHz)\n*************************************\n")
            start = float(input("start point: "))
            stop = float(input("end point: "))
            fm_band(sdr, start, stop)

        except ValueError:
            print("ERROR: input values must be float!")


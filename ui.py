# author Qi Luo
# parking_simulator_UI   visualize parking system
import Tkinter as tki
from Tkinter import *
import pandas as pds
import numpy as np
import random as rdm


# basic queueing model
class queue:
    def __init__(self, input_value):
        self.starting_time = input_value[0]    # starting time of queueing
        self.ending_time = input_value[1]    # ending time of queueing
        self.lambda_value = input_value[2]    # lambda for Poisson distribution
        self.prob = input_value[3]    # probability for probe cars

    def generate1(self):
        sum_t = 0
        t_list = []
        while (sum_t < self.ending_time - self.starting_time):
            t_i = rdm.expovariate(self.lambda_value)
            sum_t = sum_t + t_i
            t_list.append(sum_t)
        return t_list

    def generate2(self):
        sum_t = 0
        t_list = []
        while (sum_t < self.ending_time - self.starting_time):
            t_i = rdm.expovariate(self.lambda_value)
            sum_t = sum_t + t_i
            t_list.append(t_i)
        return t_list

    def counting(self, input_list):
        return len(input_list)

    def classify(self, count):
        type1_count = np.random.binomial(count, self.prob)
        return type1_count, (count - type1_count)


class Tkqueue:
    def __init__(self, input_time1, input_time2, bitmap=None):
        self.tk = tk = Tk()
        self.entering_time = input_time1
        self.leaving_time = input_time2
        self.num_cars = len(input_time1)
        self.canvas = c = Canvas(tk)
        c.pack()
        width, height = tk.getint(c['width']), tk.getint(c['height'])
        # add background bitmap
        if bitmap:
            self.bitmap = c.create_bitmap(width / 2, height / 2,
                {'bitmap': bitmap, 'foreground': 'blue'})


def counting_cars(entering_input, parking_input):
    # entering queueing
    entering = queue(entering_input)
    entering_time = np.array(entering.generate1())
    count_entering = entering.counting(entering_time)
    parking = queue(parking_input)
    parking_time = np.array(parking.generate2())
    leaving_time = entering_time + parking_time[0: count_entering]
    [type1_count, type2_count] = entering.classify(count_entering)
    return_vector = [entering_time, leaving_time, type1_count, type2_count]
    return return_vector


def main():
    entering_input = [0, 100, 1, 0.5]
    parking_input = [0, 2000, 0.1, 0.5]
    output_1 = counting_cars(entering_input, parking_input)
    [entering_time, leaving_time, type1_count, type2_count] = output_1
    print(output_1)

if __name__ == '__main__':
    main()
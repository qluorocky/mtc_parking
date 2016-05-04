# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:27:23 2016

@author: qi
"""

from kivy.app import App
from kivy.uix.widget import Widget

class Probe_Car(Widget):
    pass

class CarApp(App):
    def build(self):
        return Probe_Car()
        
if __name__ == '__main__':
    CarApp().run()
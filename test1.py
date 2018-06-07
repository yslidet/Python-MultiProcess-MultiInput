# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:44:04 2018

@author: lidet

>> Traditional Threading 
- Standard Producer/Consumer Threading Pattern
"""

import time
import threading 
import Queue

class Consumer(threading.Thread):
    def __init__(self,queue):
        threading.Thread.__init__(self)
        self._queue = queue
        
    def run(self):
        while True: 
            #queue.get() block the current thread until an item is retrieved
            msg = self._queue.get()
            


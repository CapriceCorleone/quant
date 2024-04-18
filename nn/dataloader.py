'''
Author: WangXiang
Date: 2024-04-18 21:45:34
LastEditTime: 2024-04-18 21:51:45
'''

from typing import Optional
from threading import Thread
from queue import Empty, Full, Queue
from multiprocessing.context import SpawnContext

import torch
from torch.utils.data import Dataset, DataLoader


class ThreadBuffer:
    """
    Iterates over values from self.src in a separate thread but yielding them in the current thread. This allows values
    to be queued up asynchronously. The internal thread will continue running so long as the source has values or until
    the stop() method is called.

    One issue raised by using a thread in this way is that during the lifetime of the thread the source object is being
    iterated over, so if the thread hasn't finished another attempt to iterate over it will raise an exception or yield
    unexpected results. To ensure the thread releases the iteration and proper cleanup is done the stop() method must
    be called which will join with the thread.

    Args:
        src: Souce data iterable
        buffer_size: Number of items to buffer from the source
        timeout: Time to wait for an item from the buffer, or to wait while the buffer is full when adding items
    """

    def __init__(self, src, buffer_size: int = 1, timeout: float = 0.01) -> None:
        self.src = src
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.buffer: Queue = Queue(self.buffer_size)
        self.gen_thread = Optional[Thread] = None
        self.is_running = False

    def enqueue_values(self):
        for src_val in self.src:
            while self.is_running:
                try:
                    self.buffer.put(src_val, timeout=self.timeout)
                except Full:
                    pass  # try to add the item again
                else:
                    break  # successfully added the item, quit trying
from fish_tracking.common.math_utils import *
class ATDBuffer:
    def __init__(self, size):
        self.size = size
        self.current_counter = 0
        self.atd_buffer = np.zeros((size,3))
        self.value = np.array([0.0,0.0,0.0])

    def get_value(self):
        return self.value

    def update(self, new_value):
        self.atd_buffer[self.current_counter] = new_value
        self.current_counter = 0 if self.current_counter == self.size - 1 else self.current_counter + 1
        r = np.array([0.0,0.0,0.0])
        for i in range(self.size):
            r = r+ self.atd_buffer[i]
        self.value = r / self.size

        return 0

    def reset(self):
        self.value = np.array([0.0,0.0,0.0])
        self.atd_buffer = np.zeros((self.size,3))
        return 0

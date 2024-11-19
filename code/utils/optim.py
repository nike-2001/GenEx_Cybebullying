# -*- coding: utf-8 -*-
# This script defines a ScheduledOptim class to manage learning rate scheduling during optimization.

import math
import numpy as np


class ScheduledOptim:
    '''
    A wrapper class for learning rate scheduling.
    This class dynamically adjusts the learning rate based on the number of steps taken during optimization.
    '''

    def __init__(self, optimizer, lr, decay_step=1000, decay_rate=0.9, steps=0):
        """
        Initialize the ScheduledOptim object.

        Args:
            optimizer: The optimizer instance (e.g., Adam, SGD) to be wrapped.
            lr (float): The initial learning rate.
            decay_step (int): The number of steps before the learning rate is decayed.
            decay_rate (float): The factor by which the learning rate is multiplied during decay.
            steps (int): Initial step count, used for resuming training or scheduling.
        """
        self.init_lr = lr  # Store the initial learning rate
        self.steps = steps  # Initialize the current step counter
        self._optimizer = optimizer  # Store the optimizer instance
        self.decay_rate = decay_rate  # Learning rate decay factor
        self.decay_step = decay_step  # Step interval for decaying the learning rate

    def step(self):
        """
        Perform a step of optimization while updating the learning rate.
        """
        self._update_learning_rate()  # Update the learning rate based on the current step
        self._optimizer.step()  # Perform the optimization step using the wrapped optimizer

    def zero_grad(self):
        """
        Reset the gradients of all model parameters.
        """
        self._optimizer.zero_grad()  # Clear the gradients in the wrapped optimizer

    def _update_learning_rate(self):
        """
        Adjust the learning rate based on the number of steps taken.
        Implements exponential decay of the learning rate.
        """
        self.steps += 1  # Increment the step counter
        if self.steps >= self.decay_step:
            # Calculate the new learning rate with exponential decay
            lr = self.init_lr * math.pow(self.decay_rate, int(self.steps / self.decay_step))
            # Update the learning rate for each parameter group in the optimizer
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Use the initial learning rate before the decay step is reached
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = self.init_lr

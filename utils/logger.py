from __future__ import absolute_import
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['Logger', 'LoggerMonitor', 'savefig', 'closefig']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)

def closefig():
    plt.close()


def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title is None else title
        self.column_width = 12  # Set a fixed column width for both names and numbers
        
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {name.strip(): [] for name in self.names}

                for line in self.file:
                    values = line.rstrip().split('\t')
                    for i in range(len(values)):
                        value = values[i].strip()  # Remove any extra spaces
                        self.numbers[self.names[i].strip()].append(float(value) if value != 'None' else None)
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            return  # No need to set names again if resuming
        self.names = names
        self.numbers = {name: [] for name in self.names}
        
        # Write header with fixed column width and left alignment
        formatted_names = [name.ljust(self.column_width) for name in self.names]
        header = '\t'.join(formatted_names) + '\n'
        self.file.write(header)
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        
        # Format numbers to have fixed width and left alignment
        formatted_numbers = []
        for num in numbers:
            if num is not None:
                formatted_numbers.append("{0:.5f}".format(num).ljust(self.column_width))
            else:
                formatted_numbers.append('None'.ljust(self.column_width))
        line = '\t'.join(formatted_numbers) + '\n'
        self.file.write(line)
        self.file.flush()

        # Append numbers to internal storage
        for index, num in enumerate(numbers):
            self.numbers[self.names[index]].append(num)

    def plot(self, names=None):
        names = self.names if names is None else names
        for name in names:
            x = np.arange(len(self.numbers[name]))
            plt.plot(x, np.asarray(self.numbers[name]), label=self.title + '(' + name + ')')
        plt.legend()
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)
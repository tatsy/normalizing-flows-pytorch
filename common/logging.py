import os
import sys

import colorama
from colorama import Fore


class Logging(object):
    """
    Logging class with colored message types
    """
    def __init__(self, name=None):
        colorama.init(autoreset=True)

    def print(self, msg):
        sys.stdout.write(Fore.RESET + msg + os.linesep)

    def _format(self, color, tag, msg):
        tag_str = '[ ' + tag + ' ] '
        tag_len = len(tag_str)
        msg_lines = msg.split('\n')
        line_1st = color + tag_str + Fore.RESET + msg_lines[0] + os.linesep
        line_rest = [(' ' * tag_len + l + os.linesep) for l in msg_lines[1:]]
        sys.stdout.write(line_1st + ''.join(line_rest))

    def debug(self, msg):
        self._format(Fore.GREEN, 'DEBUG', msg)

    def info(self, msg):
        self._format(Fore.BLUE, 'INFO', msg)

    def warning(self, msg):
        self._format(Fore.YELLOW, 'WARN', msg)

    def error(self, msg):
        self._format(Fore.RED, 'ERROR', msg)

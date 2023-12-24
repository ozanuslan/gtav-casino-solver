#!/bin/python3

from pynput.keyboard import Key, Listener, HotKey, KeyCode
import os
import time
from multiprocessing import Process, Queue
import pyscreeze
import uinput
import subprocess

def inclusive_range(start, end, step=1):
    return range(start, end + step, step)


def log(message, *args, **kwargs):
    message = str(message)
    _sep = kwargs.get('sep', ' ')
    message = message + _sep.join(args)
    _end = kwargs.get('end', '')
    message = message + _end
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('[', date, '] ', message, sep='')


class FingerprintSolver:

    screenshot = None

    def __init__(self, fp_dir, keyboard_wrapper):
        self.fp_dir = fp_dir
        self.file_extension = '.png'
        self.main_fp_suffix = '_main' + self.file_extension
        self.main_fp_dict = self.__get_main_fp_dict()
        self.solution_fp_dict = self.__get_solution_fp_dict()
        self.keyb = keyboard_wrapper

    def __get_fingerprint_path(self, fp_name):
        return os.path.join(self.fp_dir, fp_name)

    def __get_main_fp_dict(self):
        main_fp_dict = {}
        for fp_name in os.listdir(self.fp_dir):
            if fp_name.endswith(self.main_fp_suffix):
                main_fp_dict[fp_name[:-len(self.main_fp_suffix)]
                             ] = self.__get_fingerprint_path(fp_name)
        return main_fp_dict

    def __get_solution_fp_dict(self):
        solution_fp_dict = {}
        for fp_name in os.listdir(self.fp_dir):
            for i in inclusive_range(1, 4):
                suffix = '_{}.png'.format(i)
                if fp_name.endswith(suffix):
                    base_fp_name = fp_name[:-len(suffix)]
                    if solution_fp_dict.get(base_fp_name) is None:
                        solution_fp_dict[base_fp_name] = []
                    solution_fp_dict[base_fp_name].append(
                        self.__get_fingerprint_path(fp_name))
        return solution_fp_dict

    def __get_matching_base_fp(self):
        q = Queue()
        processes = []
        self.screenshot = pyscreeze.screenshot()
        for main_fp in self.main_fp_dict.values():
            p = Process(target=self.__get_matching_fp_helper,
                        args=(main_fp, self.screenshot, q))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        results = [q.get() for p in processes]
        results = [x for x in results if x is not None]

        if len(results) > 1:
            raise Exception('More than one main fingerprint found')
        elif len(results) == 0:
            raise Exception('No main fingerprint found')
        else:
            return results[0]

    def __get_matching_fp_helper(self, fp_path, screenshot, q):
        try:
            box = pyscreeze.locate(fp_path, screenshot, grayscale=True, confidence=0.8)
        except Exception as e:
            log(e)
            box = None

        if box is not None:
            q.put((self.__get_base_fp_name(fp_path), box))
        else:
            q.put(None)

    def __get_matching_solution_fps(self, base_fp_name):
        q = Queue()
        processes = []
        for solution_fp in self.solution_fp_dict[base_fp_name]:
            p = Process(target=self.__get_matching_fp_helper,
                        args=(solution_fp, self.screenshot, q))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        results = [q.get() for p in processes]
        results = [x for x in results if x is not None]
        return results

    def __get_base_fp_name(self, base_fp_path):
        if not base_fp_path.endswith(self.main_fp_suffix):
            return os.path.basename(base_fp_path)[:-(len(self.file_extension))]
        return os.path.basename(base_fp_path)[:-len(self.main_fp_suffix)]

    def solve_fingerprints(self):
        log('Looking for main fingerprint')
        try:
            res = self.__get_matching_base_fp()
        except Exception as e:
            log(e)
            return False
        log('Found main fingerprint: {}'.format(res), sep='')
        base_fp_name, box = res
        log('Looking for solution fingerprints')
        solution_fps = self.__get_matching_solution_fps(base_fp_name)
        if len(solution_fps) < 4:
            log('Not enough solution fingerprints found ({} found, 4 needed)'.format(
                len(solution_fps)))
            return False

        solution = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ]
        for solution_fp in solution_fps:
            _fp, box = solution_fp
            x, y = box.left, box.top
            # mod x and y to be in width and height
            x = x % width
            y = y % height
            # get the index of the x and y
            x_index = self.__which_x_index(x)
            y_index = self.__which_y_index(y)

            if x_index is None or y_index is None:
                log('Invalid solution fingerprint location found: {}'.format(box))
                return False

            solution[y_index][x_index] = 1

        log('Found solution fingerprints: {}'.format(solution))

        self.__input_solution(solution)

        return True

    def __input_solution(self, solution):
        solution_inputter_path = os.path.join(
            os.path.dirname(__file__), 'solution_inputter')
        subprocess.run([solution_inputter_path, str(solution)])

# saygilar by sono

    def __which_y_index(self, y):
        if y < 285:
            return 0
        elif y < 440:
            return 1
        elif y < 595:
            return 2
        elif y < 750:
            return 3
        else:
            return None

    def __which_x_index(self, x):
        if x < 485:
            return 0
        elif x < 635:
            return 1
        else:
            return None


def fingerprint_solve_event():
    start = time.time()
    success = fingerprintsolver.solve_fingerprints()
    end = time.time()
    dur = end - start
    log('Took: {:.2f}s'.format(dur))
    if success:
        log('Solved all fingerprints')
    else:
        log('Failed to solve fingerprints')


class Kb:
    W = uinput.KEY_W
    A = uinput.KEY_A
    S = uinput.KEY_S
    D = uinput.KEY_D
    TAB = uinput.KEY_TAB
    ENTER = uinput.KEY_ENTER

    def __init__(self):
        self.keyb = uinput.Device(
            [self.W, self.A, self.S, self.D, self.TAB, self.ENTER])
        super()

    def send(self, key):
        log('Sending {}'.format(key))
        self.keyb.emit_click(key)
        time.sleep(0.025)


class HotkeyExecutor:
    def __init__(self, hotkeys):
        self.hotkeys = hotkeys

    def signal_press_to_hotkeys(self, key):
        for hotkey in self.hotkeys:
            hotkey.press(listen.canonical(key))

    def signal_release_to_hotkeys(self, key):
        for hotkey in self.hotkeys:
            hotkey.release(listen.canonical(key))

    def prepare_hotkeys(self):
        global listen
        with Listener(on_press=self.signal_press_to_hotkeys, on_release=self.signal_release_to_hotkeys) as listen:
            listen.join()


def bye():
    log('Exiting. Bye!')
    exit(0)


def main():
    log('Setting up...')
    global width, height
    width = 1920
    height = 1080
    # get script base dir
    fp_dir_name = 'fp_{}_{}'.format(width, height)
    base_path = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(base_path, fp_dir_name)

    keyboard_wrapper = Kb()

    global fingerprintsolver
    fingerprintsolver = FingerprintSolver(full_path, keyboard_wrapper)

    CTRL = Key.ctrl
    CTRL_E = [CTRL, KeyCode.from_char('e')]
    CTRL_Q = [CTRL, KeyCode.from_char('q')]

    fp_hotkey = HotKey(CTRL_E, fingerprint_solve_event)
    exit_hotkey = HotKey(CTRL_Q, bye)

    hotkeys = [fp_hotkey, exit_hotkey]
    hke = HotkeyExecutor(hotkeys)

    log('Ready! Press {} to solve fingerprints'.format(CTRL_E))
    log('Press {} to exit'.format(CTRL_Q))
    hke.prepare_hotkeys()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        bye()

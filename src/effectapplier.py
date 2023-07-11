import os
import pedalboard as pdb
import numpy as np
import torchaudio
import torch
import math
class EffectApplier():

    def __init__(self, input_dir, input_name, output_dir=None, output_name=None):
        self.audio, self.sr = torchaudio.load(os.path.join(input_dir, input_name))
        self.audio = self.audio / abs(self.audio).max()
        if output_dir is None:
            self.output_dir = input_dir
        else:
            self.output_dir = output_dir
        if output_name is None:
            self.output_name = input_name + '_processed'
        else:
            self.output_name = output_name
        self.od = -1
        self.cs = -1
        self.tr = -1
        self.dl = -1
        self.rv = -1
        self.od_v = -1
        self.cs_v = -1
        self.tr_v = -1
        self.dl_v = -1
        self.rv_v = -1
        self.board = pdb.Pedalboard()

    def addDistortion(self, val, mode='random'):
        if mode == 'random':
            gain = self.__randomize(val, val + 0.2, gain=100.0)
        else:
            gain = val * 100.0

        # vst2 = pdb.load_plugin("_assets/Dystortion.vst3")
        # vst2.drive = round(5.0 + gain * 5.0, 2)
        # self.board.append(vst2)

        gain_dB = self.__nonlinearCorrection(gain, minv=0, maxv=50, coeff=1)
        # gain_dB = round(gain, 2)
        self.board.append(pdb.Clipping(threshold_db=-3.0))
        self.board.append(pdb.Distortion(drive_db=gain_dB))

        self.od = int(val / 0.2)
        self.od_v = round(gain / 100.0, 2)
        return gain

    def addChorus(self, val, mode='random'):
        if mode == 'random':
            depth = self.__randomize(val, val + 0.2)
        else:
            depth = val

        self.board.append(pdb.Chorus(rate_hz=0.5, depth=depth,
                                     centre_delay_ms=20.0, feedback=0.5, mix=0.5))

        self.cs = int(val / 0.2)
        self.cs_v = depth
        return depth

    def addTremolo(self, val, mode='random'):
        vst = pdb.load_plugin("_assets/TaS-X.vst3")

        if mode == 'random':
            rate = self.__randomize(val, val + 0.2, gain=10.0)
        else:
            rate = val * 10.0

        vst.depth = 1.0
        vst.dry = 0.1
        vst.wet = 0.9
        vst.rate = rate
        self.board.append(vst)

        self.tr = int(val / 0.2)
        self.tr_v = round(rate / 10.0, 2)
        return rate

    # def addPhaser(self, val, mode='random'):
    #     if mode == 'random':
    #         depth = self.__randomize(val, val + 0.2)

    #     self.tr = int(val / 0.2)

    #     self.board.append(pdb.Phaser(rate_hz=2.0, depth=depth,
    #                                  centre_frequency_hz=800.0, feedback=0.5, mix=0.5))
    #     self.tr_v = depth
    #     return depth

    def addDelay(self, val, mode='random'):
        if mode == 'random':
            time = self.__randomize(val, val + 0.2)
        else:
            time = val

        self.board.append(pdb.Delay(delay_seconds=time + 0.02, feedback=0.5, mix=0.5))

        self.dl = int(val / 0.2)
        self.dl_v = time
        return time

    def addReverb(self, val, mode='random'):
        if mode == 'random':
            decay = self.__randomize(val, val + 0.2)
        else:
            decay = val

        self.board.append(pdb.Reverb(room_size=decay, damping=0.1, wet_level=0.75,
                                     dry_level=0.25, width=1.0, freeze_mode=0.0))

        self.rv = int(val / 0.2)
        self.rv_v = decay
        return decay

    def addEffect(self, fx, val, mode='random'):
        if (fx == 0):
            self.addDistortion(val, mode)
        elif (fx == 1):
            self.addChorus(val, mode)
        elif (fx == 2):
            self.addTremolo(val, mode)
        elif (fx == 3):
            self.addDelay(val, mode)
        elif (fx == 4):
            self.addReverb(val, mode)

    def generate(self):
        effected = self.board(self.audio.numpy(), self.sr)

        waveform = torch.from_numpy(effected)
        waveform = waveform / abs(waveform).max()

        # now = torchaudio.functional.loudness(waveform, self.sr)
        # diff = torchaudio.functional.DB_to_amplitude(now + 24, 1, 0.5)
        # waveform = waveform / diff.item()
        # if abs(waveform).max() > 1:
        #     print("clipping")

        return waveform

    def save(self):
        signal = self.generate()

        output_name = self.output_name

        if self.od > -1:
            output_name += "_od_%s" % self.od
            od_bool = 1
        else:
            od_bool = 0
        if self.cs > -1:
            output_name += "_cs_%s" % self.cs
            cs_bool = 1
        else:
            cs_bool = 0
        if self.tr > -1:
            output_name += "_tr_%s" % self.tr
            tr_bool = 1
        else:
            tr_bool = 0
        if self.dl > -1:
            output_name += "_dl_%s" % self.dl
            dl_bool = 1
        else:
            dl_bool = 0
        if self.rv > -1:
            output_name += "_rv_%s" % self.rv
            rv_bool = 1
        else:
            rv_bool = 0

        output_path = os.path.join(self.output_dir, output_name + ".wav")

        torchaudio.save(output_path,
                        signal,
                        self.sr,
                        encoding="PCM_S",
                        bits_per_sample=16,
                        format="wav")
        return [output_name,
                od_bool, cs_bool, tr_bool, dl_bool, rv_bool,
                self.od, self.cs, self.tr, self.dl, self.rv,
                self.od_v, self.cs_v, self.tr_v, self.dl_v, self.rv_v]

    def reset(self):
        self.od = 0
        self.cs = 0
        self.tr = 0
        self.dl = 0
        self.rv = 0
        self.od_v = -1
        self.cs_v = -1
        self.tr_v = -1
        self.dl_v = -1
        self.rv_v = -1
        self.board = pdb.Pedalboard()

    def __randomize(self, min, max, gain=1):
        val = min - 1
        while val <= min or val >= max:
            val = np.random.normal((min + max) / 2, 0.025, 1)
            val = val[0]
        return round(val * gain, 2)

    def __nonlinearCorrection(self, value, minv, maxv, coeff):
        return round(minv + (maxv - minv) * math.pow(value / 100.0, coeff), 2)

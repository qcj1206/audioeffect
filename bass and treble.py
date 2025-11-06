import os
import soundfile as sf
import scipy.signal as sg
import numpy as np

# ---------- 工具宏 ----------
M_PI   = np.pi
DB_TO_LINEAR = lambda x: 10.0 ** (x * 0.05)

# ---------- 1. 系数计算 ----------
def Coefficients(hz: float, slope: float, gain: float, samplerate: float,
                 type_: int,
                 a0: float, a1: float, a2: float,
                 b0: float, b1: float, b2: float):
    """
    type_  : 0=kBass  1=kTreble
    """
    w = 2.0 * M_PI * hz / samplerate
    a = np.exp(np.log(10.0) * gain / 40.0)
    b = np.sqrt((a * a + 1) / slope - (a - 1) ** 2)

    if type_ == 0:  # kBass
        b0 = a * ((a + 1) - (a - 1) * np.cos(w) + b * np.sin(w))
        b1 = 2 * a * ((a - 1) - (a + 1) * np.cos(w))
        b2 = a * ((a + 1) - (a - 1) * np.cos(w) - b * np.sin(w))
        a0 = ((a + 1) + (a - 1) * np.cos(w) + b * np.sin(w))
        a1 = -2 * ((a - 1) + (a + 1) * np.cos(w))
        a2 = ((a + 1) + (a - 1) * np.cos(w) - b * np.sin(w))
    else:  # kTreble
        b0 = a * ((a + 1) + (a - 1) * np.cos(w) + b * np.sin(w))
        b1 = -2 * a * ((a - 1) + (a + 1) * np.cos(w))
        b2 = a * ((a + 1) + (a - 1) * np.cos(w) - b * np.sin(w))
        a0 = ((a + 1) - (a - 1) * np.cos(w) + b * np.sin(w))
        a1 = 2 * ((a - 1) - (a + 1) * np.cos(w))
        a2 = ((a + 1) - (a - 1) * np.cos(w) - b * np.sin(w))

    return (b0 / a0, b1 / a0, b2 / a0, a0 / a0, a1 / a0, a2 / a0)


# ---------- 2. 状态机 ----------
class BassTrebleState:
    __slots__ = (
        'samplerate', 'slope', 'hzBass', 'hzTreble',
        'a0Bass', 'a1Bass', 'a2Bass', 'b0Bass', 'b1Bass', 'b2Bass',
        'a0Treble', 'a1Treble', 'a2Treble', 'b0Treble', 'b1Treble', 'b2Treble',
        'xn1Bass', 'xn2Bass', 'yn1Bass', 'yn2Bass',
        'xn1Treble', 'xn2Treble', 'yn1Treble', 'yn2Treble',
        'bass', 'treble', 'gain'
    )

    def __init__(self):
        self.samplerate = 48000.0
        self.slope = 0.4
        self.hzBass = 250.0
        self.hzTreble = 4000.0
        # 系数区
        self.a0Bass = self.a1Bass = self.a2Bass = 1.0
        self.b0Bass = self.b1Bass = self.b2Bass = 0.0
        self.a0Treble = self.a1Treble = self.a2Treble = 1.0
        self.b0Treble = self.b1Treble = self.b2Treble = 0.0
        # 延迟线
        self.xn1Bass = self.xn2Bass = self.yn1Bass = self.yn2Bass = 0.0
        self.xn1Treble = self.xn2Treble = self.yn1Treble = self.yn2Treble = 0.0
        # 缓存旧值
        self.bass = -1.0
        self.treble = -1.0
        self.gain = 1.0


# ---------- 3. DoFilter ----------
def DoFilter(st: BassTrebleState, in_: float) -> float:
    # Bass shelf
    out = (st.b0Bass * in_ + st.b1Bass * st.xn1Bass + st.b2Bass * st.xn2Bass
           - st.a1Bass * st.yn1Bass - st.a2Bass * st.yn2Bass) / st.a0Bass
    st.xn2Bass = st.xn1Bass
    st.xn1Bass = in_
    st.yn2Bass = st.yn1Bass
    st.yn1Bass = out

    # Treble shelf
    in_ = out
    out = (st.b0Treble * in_ + st.b1Treble * st.xn1Treble + st.b2Treble * st.xn2Treble
           - st.a1Treble * st.yn1Treble - st.a2Treble * st.yn2Treble) / st.a0Treble
    st.xn2Treble = st.xn1Treble
    st.xn1Treble = in_
    st.yn2Treble = st.yn1Treble
    st.yn1Treble = out

    return out


# ---------- 4. InstanceInit ----------
def InstanceInit(st: BassTrebleState, sampleRate: float,
                 bass_db: float, treble_db: float, gain_db: float):
    st.samplerate = sampleRate
    st.slope = 0.4
    st.hzBass = 250.0
    st.hzTreble = 4000.0

    # 清零延迟线
    st.xn1Bass = st.xn2Bass = st.yn1Bass = st.yn2Bass = 0.0
    st.xn1Treble = st.xn2Treble = st.yn1Treble = st.yn2Treble = 0.0

    # 系数占位
    st.a0Bass = st.a0Treble = 1.0
    st.a1Bass = st.a2Bass = st.b0Bass = st.b1Bass = st.b2Bass = 0.0
    st.a1Treble = st.a2Treble = st.b0Treble = st.b1Treble = st.b2Treble = 0.0

    st.bass = -1.0
    st.treble = -1.0
    st.gain = DB_TO_LINEAR(gain_db)


# ---------- 5. InstanceProcess ----------
def InstanceProcess(st: BassTrebleState,
                    bass_db: float, treble_db: float, gain_db: float,
                    inBlock: np.ndarray) -> np.ndarray:
    """
    一入一出，blockLen 任意
    """
    # 1. 缓存旧值
    oldBass = st.bass
    oldTreble = st.treble

    # 2. 计算新系数（仅当参数变化）
    if bass_db != oldBass:
        st.b0Bass, st.b1Bass, st.b2Bass, st.a0Bass, st.a1Bass, st.a2Bass = \
            Coefficients(st.hzBass, st.slope, bass_db, st.samplerate, 0,
                         st.a0Bass, st.a1Bass, st.a2Bass,
                         st.b0Bass, st.b1Bass, st.b2Bass)
        st.bass = bass_db

    if treble_db != oldTreble:
        st.b0Treble, st.b1Treble, st.b2Treble, st.a0Treble, st.a1Treble, st.a2Treble = \
            Coefficients(st.hzTreble, st.slope, treble_db, st.samplerate, 1,
                         st.a0Treble, st.a1Treble, st.a2Treble,
                         st.b0Treble, st.b1Treble, st.b2Treble)

    st.treble = treble_db
    # ---- 4. 更新总增益 ----
    st.gain = DB_TO_LINEAR(gain_db)

    # ---- 5. 逐样本滤波 ----
    outBlock = np.empty_like(inBlock)
    for i in range(inBlock.size):
        outBlock[i] = DoFilter(st, inBlock[i]) * st.gain

    return outBlock

def calculate_db(data):
    current_db = 0
    for i in range(len(data)):
        current_db += 20* np.log10(abs(data[i]))
    return current_db / (len(data))


class SoftClip:
    """tanh 软剪，可替换为 cubic/atan"""
    def __init__(self, drive_db=0.0):
        self.gain = 10 ** (drive_db / 20.0)

    def process(self, x):
        return np.tanh(x * self.gain) / self.gain



def frame_peak(x):
    return np.abs(x).max()

def db_to_linear(db):
    return 10.0 ** (db / 20.0)


filepath = "./data/"
files = os.listdir("./data/")

for file in files:
    data,fs = sf.read(filepath + file)

    block_size = 1024


    state1 = BassTrebleState()
    state2 = BassTrebleState()
    InstanceInit(state1, fs, 0.0, 0.0, 0.0)     # 初始清零
    InstanceInit(state2, fs, 0.0, 0.0, 0.0)     # 初始清零
    for i in range(int(data.shape[0] / block_size)):
        #真正处理
        # ① 先测“原始”响度（还没 EQ）

        data_l = data[i * block_size:(i + 1) * block_size, 0]
        data_r = data[i * block_size:(i + 1) * block_size, 1]

        data[i * block_size:(i+1)*block_size, 0] = InstanceProcess(state1,  15, 0.0, 0.0, data_l)  # +15 dB bass, +0 dB treble
        data[i * block_size:(i+1)*block_size, 1] = InstanceProcess(state2,  15, 0.0, 0.0, data_r)  # +15 dB bass, +0 dB treble

    data = np.clip(data,-1,1)
    #soft = SoftClip(drive_db=-0.1)
    #data = soft.process(data)
    sf.write("result/" + file.split(".")[0] + "_out.wav", data, fs)

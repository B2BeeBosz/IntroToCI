import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Initialize Universal Set {Temp, WaterVolume, RiceAmount, Time, RiceQuality}
# Inputs
Temp = ctrl.Antecedent(np.arange(0, 101, 1), 'Temperature')
WaterVolume = ctrl.Antecedent(np.arange(0, 1001, 10), 'WaterVolume')
RiceAmount = ctrl.Antecedent(np.arange(0, 1001, 10), 'RiceAmount')
Time = ctrl.Antecedent(np.arange(0, 61, 1), 'Time')
# Outputs
RiceQuality = ctrl.Consequent(np.arange(0, 101, 1), 'RiceQuality')

# Membership Function
# 1. Temperature
Temp['verylow'] = fuzz.trimf(Temp.universe, [0, 0, 20])
Temp['low'] = fuzz.trimf(Temp.universe, [15, 25, 35])
Temp['medium'] = fuzz.trimf(Temp.universe, [25, 40, 55])
Temp['high'] = fuzz.trimf(Temp.universe, [50, 65, 80])
Temp['veryhigh'] = fuzz.trimf(Temp.universe, [75, 100, 100])
# 2. WaterVolume
WaterVolume['low'] = fuzz.trapmf(WaterVolume.universe, [0, 0, 150, 250])
WaterVolume['medium'] = fuzz.trapmf(WaterVolume.universe, [150, 250, 500, 750])
WaterVolume['high'] = fuzz.trapmf(WaterVolume.universe, [500, 750, 1000, 1000])
# 3. RiceAmount
RiceAmount['low'] = fuzz.trapmf(RiceAmount.universe, [0, 0, 150, 250])
RiceAmount['medium'] = fuzz.trapmf(RiceAmount.universe, [150, 250, 500, 750])
RiceAmount['high'] = fuzz.trapmf(RiceAmount.universe, [500, 750, 1000, 1000])
# 4. Time
Time['veryshort'] = fuzz.trimf(Time.universe, [0, 0, 15])
Time['short'] = fuzz.trimf(Time.universe, [10, 17.5, 25])
Time['medium'] = fuzz.trimf(Time.universe, [20, 30, 40])
Time['long'] = fuzz.trimf(Time.universe, [35, 42.5, 50])
Time['verylong'] = fuzz.trimf(Time.universe, [45, 60, 60])
# 5. RiceQuality
RiceQuality['undercooked'] = fuzz.trimf(RiceQuality.universe, [0, 0, 25])
RiceQuality['mushy'] = fuzz.trimf(RiceQuality.universe, [20, 35, 50])
RiceQuality['tendor'] = fuzz.trimf(RiceQuality.universe, [45, 60, 75])
RiceQuality['overcooked'] = fuzz.trimf(RiceQuality.universe, [70, 100, 100])

# View Membership Function
Temp.view()
WaterVolume.view()
RiceAmount.view()
Time.view()
RiceQuality.view()
plt.show()

# Fuzzy Inference Rules
# rule1 = ctrl.Rule()
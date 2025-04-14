from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.sim_engine import SimObj, batch_sim
from datetime import timedelta, datetime

# 创建模拟环境
start_time = datetime.now()
patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')
scenario = RandomScenario(start_time=start_time, seed=1)
env = T1DSimEnv(patient, sensor, pump, scenario)

# 创建控制器
controller = BBController()

# 创建模拟对象
s1 = SimObj(env, controller, timedelta(days=1), animate=False, path='./results')

# 运行批量模拟
results = batch_sim([s1], parallel=True)
print(results)
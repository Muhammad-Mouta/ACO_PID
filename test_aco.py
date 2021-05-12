import control
import aco_pid_tuning as aco

s = control.TransferFunction.s
g = 1 / (s * (s+1) * (s+5))

aco.aco_pid_tune(g, 100, 50, 100)

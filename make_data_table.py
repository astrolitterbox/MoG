import numpy as np

data = np.genfromtxt("data.dat", delimiter=",")

califa_id = data[:, 0]
#log_vel = 100*np.log10(np.abs(data[:, 1]))
#lum = 100*data[:, 2]
#lum_err = 10*np.abs(data[:, 3]) + 2
#vel_err = 100*np.abs(np.log10(np.abs(data[:, 4]))) + 2

log_vel = np.log10(np.abs(data[:, 1]))

lum_err = np.abs(data[:, 3]) 
vel_err = np.abs(np.log10(0.4*np.abs(data[:, 4]))) 
lum = data[:, 2]

out = np.zeros((data.shape[0], 7))
out[:, 0] = califa_id
out[:, 1] = log_vel
out[:, 2] = lum
out[:, 3] = lum_err
out[:, 4] = vel_err

np.savetxt("data_allerr.dat", out, delimiter=",", fmt='%f')

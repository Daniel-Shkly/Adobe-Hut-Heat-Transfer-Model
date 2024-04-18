# By: Daniel Shklyarman
#     100851439

# A simplified model of heat transfer
# in an adobe hut

# The adobe hut is made out of brick and has no windows
# It is assumed to be small enough that it always receives
# an equal amount of sunlight/heat on all sides so that there
# is no angular dependence

# Imports
import numpy as np
import math
import matplotlib.pyplot as plt
import time

# Physical Constants

# | Name    | Description                                              | Value   | Units         |
# |---------|----------------------------------------------------------|---------|---------------|
# | k_wall  | thermal conductivity of brick                            | 0.75    | W m^-1 K^-1   |
# | k_air   | thermal conductivity of air                              | 4e-2    | W m^-1 K^-1   |
# | rho_wall| density of the wall/brick                                | 1.8e+3  | kg m^-3       |
# | rho_air | density of the air                                       | 1.2     | kg m^-3       |
# | cp_wall | heat capacity of the wall/brick                          | 840     | J kg^-1 K^-1  |
# | cp_air  | heat capacity of the air                                 | 700     | J kg^-1 K^-1  |
# | R_b     | radius of the hut (from center to external edge of wall) | 3.0     | m             |
# | R_a     | radius of the hut (from center to internal edge of wall) | 2.7     | m             |
# | h       | heat transfer coefficient of brick                       | 2.0     | W  m^-2  K^-1 |
# | tau     | time scale (12 hours)                                    | 4.32e+4 | s             |
# |---------|----------------------------------------------------------|---------|---------------|

# number of 'lines' through the hut
n = 50

# Parameters

# P1 = [k/(rho*cp)]_wall * [tau/(R_b)^2]
P1 = 2.4e-3 
# P2 = [k/(rho*cp)]_air * [tau/(R_b)^2]
P2 = 0.23 
# P3 = R_a / R_b
P3 = 0.9
# P4 = k_air / k_wall
P4 = 5.3e-2 
# P5 = h * R_b / k_wall
P5 = 8

# Width of a 'line'
# r0 = 1
# r0 - (N-1)*delta_r = r_(N-1) = 0
h = 1/(n-1)

# External Temperature as a function of time
def u_ext(t):
    return max(0, np.sin(np.pi * t))

# Array of 'lines'
# the phi at Phi[0] is the external edge
# progressive phi's go closer to the center
Phi = np.zeros(n)

# Set temperatures at time = 0
T_init = 0
for i in range(len(Phi)):
    Phi[i] = T_init

it_number = 10000
Phi_history = np.zeros([n,it_number-1])

def dphi_dt(temp_vec, n, i, parameter):
    temp_i_minus_1 = temp_vec[0]
    temp_i = temp_vec[1]
    temp_i_plus_1 = temp_vec[2]

    curr_r = ((n-1-i)*h)**2

    dphi_dt = (parameter) * (((n-1)**2)*(temp_i_minus_1 - 2*temp_i + temp_i_plus_1) + ((n-1)/curr_r)*(temp_i_plus_1 - temp_i_minus_1))

    return dphi_dt

# Runge-Kutta method, only returns the increment not
# the full new estimate, hence the '+=' in the loop
def Runge_Kutta(temp_vec, dphi_dt, h, i, n, parameter):
    k1 = dphi_dt(temp_vec, n, i, parameter)

    k2_temp_vec = temp_vec + [temp*(h*k1)/2 for temp in temp_vec]
    k2 = dphi_dt(k2_temp_vec, n, i, parameter)

    k3_temp_vec = temp_vec + [temp*(h*k2)/2 for temp in temp_vec]
    k3 = dphi_dt(k3_temp_vec, n, i, parameter)

    k4_temp_vec = temp_vec + [temp*(h*k3) for temp in temp_vec]
    k4 = dphi_dt(k4_temp_vec, n, i, parameter)

    return (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# Plotting stuff
# x = np.linspace(0,50,50)
# y = Phi

# plt.ion()

# fig,ax = plt.subplots(figsize=(10,5))
# line, = ax.plot(x,y)

# plt.title("Heat in an Adobe Hut")

# plt.xlabel("Lines from the outside in")
# plt.ylabel("Temperature")

# plt.ylim(-1, 2)

# The actual fancy stuff
for t in range(1, it_number):
    for i in range(len(Phi)):
        # BC2
        if i == 0:
            # k1 = P1*(2)*P5* (Phi[0] - u_ext(t/48))
            # k2 = P1*(2)*P5* (Phi[0]+(Phi[0]*k1*h)/2 - u_ext(t/48 + 1/96))
            # k3 = P1*(2)*P5* (Phi[0]+(Phi[0]*k2*h)/2 - u_ext(t/48 + 1/96))
            # k4 = P1*(2)*P5* (Phi[0]+(Phi[0]*k3*h)   - u_ext(t/48 + 1/48))

            # Phi[i] += (h/6) * (k1 + 2*k2 + 2*k3 + k4)

            Phi[i] = u_ext(t/1000) + (u_ext(t/1000) - Phi[1])/(2*P5*h)
            Phi_history[i,t-1] = Phi[i]
        
        # Heat equation in wall
        elif i < n*P3:
            temp_vec = [Phi[i-1],Phi[i],Phi[i+1]]
            Phi[i] += Runge_Kutta(temp_vec, dphi_dt , h, i, n, P1) # should be += Runge-kutta of the current expression (or something like that)
            Phi_history[i,t-1] = Phi[i]

        # BC1/4
        elif i == (n - n*P3):
            Phi[i] = (P4*Phi[i+1] - Phi[i-1]) / (P4-1)
            # dPhi_dt = Phi[i-1] 
            # Phi[i] += Runge_Kutta(Phi[i], dPhi_dt , h) # should be += Runge-kutta of the current expression (or something like that)
            Phi_history[i,t-1] = Phi[i]
        
        # Heat equation in the air
        elif i > n-n*P3 and i < n-1:
            temp_vec = [Phi[i-1],Phi[i],Phi[i+1]]
            Phi[i] += Runge_Kutta(temp_vec, dphi_dt , h, i, n, P2) # should be += Runge-kutta of the current expression (or something like that)
            Phi_history[i,t-1] = Phi[i]
        
        # BC6
        else:
            Phi[i] = Phi[i-1]
            Phi_history[i,t-1] = Phi[i]
    
    if t < 100:
        print(Phi_history[:,t-1][:5])
    
    # new_y = Phi

    # line.set_xdata(x)
    # line.set_ydata(new_y)

    # fig.canvas.draw()

    # fig.canvas.flush_events()

    # time.sleep(0.01)


# Clamping for if temperature starts going too high/low
# for i in Phi_history:
#     for j in range(len(i)):
#         if abs(i[j]) > 10:
#             i[j] = 0

# for i in range(0, n):
#     plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[i]+i, label=f'Phi{i}')

# y = np.zeros(it_number-1)
# for i in range(len(y)):
#     y[i] = n - n*P3
# plt.plot(np.linspace(0,it_number, it_number-1), y, label = 'wall', color='black')

plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[0], label=f'Phi{0}')
plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[1], label=f'Phi{1}')
plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[2], label=f'Phi{2}')
plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[3], label=f'Phi{3}')
plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[4], label=f'Phi{4}')
plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[5], label=f'Phi{5}')
plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[6], label=f'Phi{6}')
plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[7], label=f'Phi{7}')
plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[8], label=f'Phi{8}')
plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[9], label=f'Phi{9}')
plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[10], label=f'Phi{10}')
plt.plot(np.linspace(0,it_number, it_number-1), Phi_history[20], label=f'Phi{20}')

plt.legend()
plt.show()


# Boundary Conitions

# BC1 at interface of internal air and internal edge of wall

# BC2 at interface of external edge of wall and outside air

# BC3 angle stuff that we don't care about because model has no angular dependence

# BC4 temperature of internal air at interface with wall should equal temperature
# of wall at interface with internal air

# BC5 angle stuff that we don't care about because model has no angular dependence

# BC6 change in temperature wrt to radius at the center of the hut is 0



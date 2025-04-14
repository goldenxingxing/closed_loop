# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:20:00 2025

@author: liqunwei
"""

# Append result for this simulation
x=np.arange(N_iterations)*timescale/60
y=np.array(G_t)/g
plt.subplot(2, 1, 1)
plt.plot(x, np.array(G_t)/g, '--', label="Glucose Level")
plt.plot(x, np.array(G_predicted)/g, '--', label="Predicted Glucose Level")
# plt.ylabel(r"G_t, mmol/L")
plt.ylabel(r"G_t, mg/dL")
plt.legend()
plt.axhline(70.,color='red',linestyle='--',linewidth=0.75)
plt.axhline(140.,color='green',linestyle='--',linewidth=0.75)
# plt.grid()
if system.meal:
    plt.axvline(bfast_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_start*timescale/60,color='red',linestyle='--',linewidth=0.75)

    plt.axvline(bfast_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
ymax = max(np.array(G_t)/g)
xpos = np.where(y == ymax)
xmax = x[xpos]

plt.annotate(str(round(ymax,2)), xy=(xmax, ymax), xytext=(xmax, ymax + 3), arrowprops=dict(facecolor='black'),)


plt.subplot(2, 1, 2)
plt.plot(x, tau, linewidth=0,color='r',marker='.',label="Insulin Infusion")
plt.ylabel(r"I_u, mU")
plt.xlabel("Time")
plt.legend()
plt.grid()
if system.meal:
    plt.axvline(bfast_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_start*timescale/60,color='red',linestyle='--',linewidth=0.75)

    plt.axvline(bfast_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    # plt.text(bfast_start*timescale/60+0.1,4, 'Breakfast', rotation=90)
    # plt.text(lunch_start*timescale/60+0.1, 4, 'Lunch', rotation=90)
    # plt.text(dinner_start*timescale/60+0.1,4, 'Dinner', rotation=90)
plt.show()
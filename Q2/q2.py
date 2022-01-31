import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Origional four Equations
# with  k1 = 100, k2 = 600, k3 = 150
def dsdt(e, s, es):
    return ((-100 * e * s) + (600 * es))

def dedt(e, s, es):
    return ((-100 * e * s) + ((600 +150) * es))

def desdt(e, s, es):
    return ((100 * e * s) - ((600 +150) * es))

def dpdt(es):
    return (150 * es)

# fourth order runge-kutta method
# Finds value of es for a given s using step size h
# and initial value es0 at s0.
earray  = []
sarray  = []
esarray = []
parray  = []
count_array = []

def RK4(step_size, end):
    ## given information
    # initial concentation (uM)
    e0  = 1
    s0  = 10
    es0 = 0
    p0  = 0
    c   = 0
    # step size
    h = step_size
    while c <= end:
        # add the previous points
        count_array.append(c)
        earray.append(e0)
        sarray.append(s0)
        esarray.append(es0)
        parray.append(p0)
        # increment
        c += h
        # calculation
        # 1
        dsdt1  = dsdt(e0, s0, es0)
        s1    = s0 + dsdt1 * h / 2
        dedt1  = dedt(e0, s0, es0)
        e1    = e0 + dedt1 * h / 2
        desdt1 = desdt(e0, s0, es0)
        es1   = es0 + desdt1 * h / 2
        dpdt1  = dpdt(es0)
        p1    = p0 + dpdt1 * h / 2

        # 2
        dsdt2  = dsdt(e1, s1, es1)
        s2    = s0 + dsdt2 * h / 2
        dedt2  = dedt(e1, s1, es1)
        e2    = e0 + dedt2 * h / 2
        desdt2 = desdt(e1, s1, es1)
        es2   = es0 + desdt2 * h / 2
        dpdt2  = dpdt(es1)
        p2    = p0 + dpdt2 * h / 2

        # 3
        dsdt3  = dsdt(e2, s2, es2)
        s3    = s0 + dsdt3 * h
        dedt3  = dedt(e2, s2, es2)
        e3    = e0 + dedt3 * h
        desdt3 = desdt(e2, s2, es2)
        es3   = es0 + desdt3 * h
        dpdt3  = dpdt(es2)
        p3    = p0 + dpdt3 * h

        # 4
        dsdt4  = dsdt(e3, s3, es3)
        dedt4  = dedt(e3, s3, es3)
        desdt4 = desdt(e3, s3, es3)
        dpdt4  = dpdt(es3)

        # update
        s0  = s0 + (dsdt1 + 2 * dsdt2 + 2 * dsdt3 + dsdt4) * h / 6
        e0  = e0 + (dedt1 + 2 * dedt2 + 2 * dedt3 + dedt4) * h / 6
        es0 = es0 + (desdt1 + 2 * desdt2 + 2 * desdt3 + desdt4) * h / 6
        p0  = p0 + (dpdt1 + 2 * dpdt2 + 2 * dpdt3 + dpdt4) * h / 6

results = {"E" : earray,
           "S" : sarray,
           "ES" : esarray,
           "P" : parray}

RK4(0.001, 1)
df = pd.DataFrame(results)
print(df.head())
df.to_csv("Results.csv")

fig, axs = plt.subplots(2, 2)

fig.suptitle('Concentration Change of Four Species')
axs[0, 0].plot(count_array, earray)
axs[0, 0].set_title('E')
axs[0, 0].set(xlabel='Time (min)', ylabel='Concentration (uM)')
axs[0, 1].plot(count_array, sarray,  'tab:orange')
axs[0, 1].set_title('S')
axs[0, 1].set(xlabel='Time (min)', ylabel='Concentration (uM)')
axs[1, 0].plot(count_array, esarray,  'tab:red')
axs[1, 0].set_title('ES')
axs[1, 0].set(xlabel='Time (min)', ylabel='Concentration (uM)')
axs[1, 1].plot(count_array, parray, 'tab:green')
axs[1, 1].set_title('P')
axs[1, 1].set(xlabel='Time (min)', ylabel='Concentration (uM)')
plt.tight_layout()
plt.savefig("concentration_change.png")
plt.clf()


# plot V
varray = []
for es in esarray:
    varray.append(dpdt(es))
# find the maximum
max = -1
index = -1
for i in range(len(varray)):
    if varray[i] >= max:
        max = varray[i]
        index = i

plt.plot(sarray, varray)
point_info = "Vm = " + str(round(max, 2)) + "\n, S = " + str(round(sarray[index], 2))
print("Vm = " + str(round(max, 2)) + ", when S = " + str(round(sarray[index], 2)))
plt.plot(sarray[index], max, 'ro')
plt.annotate(point_info, (sarray[index], max))
plt.title("Plot of V against the Concentration of S")
plt.xlabel("Concentration of S (uM)")
plt.ylabel("Velocity V (/min)")
plt.savefig("v_plot.png")

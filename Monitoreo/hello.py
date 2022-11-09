# Import the library
import argparse #Create the parser
import datetime
parser = argparse.ArgumentParser()# Add an argument
parser.add_argument('month', type=str, default=datetime.date.today().strftime("%m"))# Parse the argument
parser.add_argument('year', type=str, default=datetime.date.today().strftime("%m"))
args = parser.parse_args() #Print "Hello" + the user input argument
print('Hello,', args.month)
print("Year",args.year)

dicta = {'JROA': {'06/2022': [72, 51, 99, 99, 80, 0, 27, 70, 26, 98, 95, 95, 85, 56, 54, 74, 49, 58, 39, 86, 96, 96]}, 'JROB': {'06/2022': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 99, 99, 74, 0, 0, 98, 93, 93, 93, 93, 93]}, 'HYOA': {'06/2022': [100, 100, 100, 100, 100, 99, 99, 99, 99, 99, 99, 100, 99, 100, 100, 100, 100, 100, 100, 100, 100, 100]}, 'HYOB': {'06/2022': [32, 92, 95, 90, 96, 90, 79, 61, 22, 45, 93, 80, 91, 94, 94, 96, 94, 94, 93, 94, 91, 92]}, 'MALA': {'06/2022': [100, 100, 100, 100, 100, 100, 100, 100, 99, 99, 99, 99, 99, 99, 99, 100, 100, 100, 100, 100, 99, 100]}, 'MERCED': {'06/2022': [100, 99, 100, 100, 99, 99, 99, 99, 97, 97, 64, 83, 0, 0, 0, 22, 99, 91, 97, 95, 91, 98]}, 'BARRANCA': {'06/2022': [43, 0, 0, 0, 0, 0, 0, 0, 67, 76, 73, 5, 0, 15, 56, 88, 7, 98, 0, 0, 1, 0]}, 'OROYA': {'06/2022': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 2, 2]}, 'JRO': {}, 'HUANCAYO': {}}


month_name = datetime.datetime.strptime(date_mes,"%m").strftime("%B")
today = datetime.datetime.today()
today_1 = today.strftime("%B")
dates = today.strftime("%Y%j")
revision = today.strftime("%c")
#print(today_1)
print(month_name)
date = date_mes+"/"+date_year
#date = "06/2022"

ruta_local = "/home/wmaster/Downloads/%s"%('JROa')


cantidad = [dicta['JROA'][date],dicta['JROB'][date],dicta['HYOA'][date],dicta['HYOB'][date],dicta['MALA'][date],dicta['MERCED'][date],dicta['BARRANCA'][date],dicta['OROYA'][date]]
print("BEFORE: ",cantidad)
#cantidad = [dicta['OROYA'][date],dicta['BARRANCA'][date],dicta['MERCED'][date],dicta['MALA'][date],  dicta['JROA'][date],dicta['JROB'][date],dicta['HYOA'][date],dicta['HYOB'][date],]
cantidad.reverse()
print("AFTER: ",cantidad)


dias= list(np.arange(0.5,num_dia))
station = list(np.arange(0.5,8))

#plt.figure(figsize=(10,4))
fig, ax = plt.subplots(figsize = (15,6))
#cmap = ListedColormap(['red', 'yellow', 'dodgerblue'])
#cmap = ListedColormap(['mediumblue','yellow','deepskyblue'])
cmap = ListedColormap(['mediumblue','mediumblue','deepskyblue'])
#cmap = ListedColormap(['mediumblue','deepskyblue'])
plt.pcolormesh(dias,station,cantidad, cmap= cmap,  edgecolors = 'k', linewidths=0.2)

#red_patch = mpatches.Patch(color="dodgerblue", label='Normal')
#red_patch1 = mpatches.Patch(color='red', label='Sin acceso a red ')
#red_patch2 = mpatches.Patch(color='yellow', label='Falla de  sistema (ADQ o PROC)')

red_patch = mpatches.Patch(color="deepskyblue", label='Normal')
#red_patch1 = mpatches.Patch(color='yellow', label='Sin acceso a red ')
red_patch2 = mpatches.Patch(color='mediumblue', label='Falla de  sistema Rx-HF')
#fig.legend(handles=[red_patch,red_patch1,red_patch2],loc = 1, borderaxespad = 0.000)
fig.legend(handles=[red_patch,red_patch2],loc = 1, borderaxespad = 0.000)

#cbar = plt.colorbar()
#cbar.set_label("Horas de Operacion", rotation = -270)
plt.title("Proyecto HF, %s-%s"%(month_name,date_year))
#ax.set_yticklabels([".","JROA","JROB","HYOA","HYOB","MALA","MERCED","BARRANCA","OROYA"])
ax.set_yticklabels([".","OROYA","BARRANCA","MERCED","MALA","HYOB","HYOA","JROB","JROA"])
#plt.xlabel("Days - %s"%(today_1))
plt.xlabel("Days - %s %s"%(month_name,date_year))
plt.ylabel("Station")
#plt.savefig("Proyecto-HF-%s"%(today_1))
plt.savefig("Proyecto-HF-%s_%s"%(month_name,date_year))

plt.show()
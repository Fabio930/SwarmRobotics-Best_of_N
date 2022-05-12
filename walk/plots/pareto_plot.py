# -*- coding: utf-8 -*-
import matplotlib.lines as mlines
import numpy as np
import os, csv, math
import matplotlib as mpl
from matplotlib import pyplot as plt

#####################################################################################################
#####################################################################################################

class Plots_pareto:
    def __init__(self):
        plt.rcParams.update({'font.size': 18})
        self.bases=[]
        self.pareto_dict={}
        self.parallel_dict={}
        self.r=[]
        self.types=["flat","binary","quad"]
        self.base = os.path.abspath("")
        self.colors=['#5ec962', '#21918c','#3b528b','#440154']
        self.par_colors=['#fde725','#21918c','#440154']
        self.styles=[':', '--','-.','-']
        self.alphas = np.linspace(0.3, 1, num=3)
        for elem in os.listdir(self.base):
            if elem[0]=='r' and elem[1]=='e' and elem[2]=='s':
                self.bases.append(os.path.join(self.base, elem))

    ##########################################################################################################################
    def fill_dict(self,path,file_name):
        K=0
        str_K=''
        for ch in file_name:
            if ch=='k':
                K=float(str_K)
                str_K=''
            str_K+=ch
        with open(os.path.join(path,file_name),newline='') as f:
            reader = csv.reader(f)
            sem_reader=0
            dict = {}
            for row in reader:
                if sem_reader==0:
                    sem_reader=1
                else:
                    if row[1] not in self.r:
                        self.r.append(row[1])
                    dict.update({(float(row[0]),row[1],int(row[2]),row[3]):(float(row[4]),float(row[5]),float(row[6]))})
        for k in dict.keys():
            if self.pareto_dict.get(k[0]):
                tmp_dic=self.pareto_dict.get(k[0])
                tmp_dic.update({(k[1],k[2],k[3]):dict.get(k)})
                self.pareto_dict.update({k[0]:tmp_dic})
            else:
                self.pareto_dict.update({k[0]:{(k[1],k[2],k[3]):dict.get(k)}})
    ##########################################################################################################################
    def print(self):
        for base in self.bases:
            self.pareto_dict={}
            self.parallel_dict={}
            for dir in os.listdir(base):
                if dir[-10:]=='resume.csv':
                    self.fill_dict(base,dir)

            for k in self.pareto_dict.keys():
                dots = [np.array([[None,None,None,None,None]]),np.array([[None,None,None,None,None]])]
                lines=[np.array([[None,None,None,None,None]]),np.array([[None,None,None,None,None]])]
                fig,ax = plt.subplots(figsize=(10, 9))

                ##########################################################################################################################
                #LABELS
                four = mlines.Line2D([], [], color='#5ec962', marker='_', linestyle='None', markeredgewidth=5, markersize=14, label='N=4')
                sixteen = mlines.Line2D([], [], color='#3b528b', marker='_', linestyle='None', markeredgewidth=5, markersize=14, label='N=16')
                flat = mlines.Line2D([], [], color='silver', marker='o', markerfacecolor='silver', linestyle='None', markeredgewidth=1.5, markersize=10, label='Flat tree')
                quad = mlines.Line2D([], [], color='silver', marker='s', markerfacecolor='silver', linestyle='None', markeredgewidth=1.5, markersize=10, label='Quad tree')
                binary = mlines.Line2D([], [], color='silver', marker='^', markerfacecolor='silver', linestyle='None', markeredgewidth=1.5, markersize=10, label='Binary tree')

                void = mlines.Line2D([], [], linestyle='None')

                r1 = mlines.Line2D([], [], color='#cfd3d7', marker='_', linestyle='None', markeredgewidth=5, markersize=14, label='r=1-Hd')
                r2 = mlines.Line2D([], [], color='#98a1a8', marker='_', linestyle='None', markeredgewidth=5, markersize=14, label='r=2(1-Hd)')
                r3 = mlines.Line2D([], [], color='#000000', marker='_', linestyle='None', markeredgewidth=5, markersize=14, label='r=3(1-Hd)')

                handles_t = [flat, quad, binary]
                handles_n = [void,four, sixteen]
                handles_r = [r1, r2, r3]
                plt.legend(handles=handles_n+handles_t+handles_r, ncol=3,loc='lower right',framealpha=.4)
                for d in self.pareto_dict.get(k).keys():
                    if d[1]!=2:
                        val=self.pareto_dict.get(k).get(d)
                        i,j=-1,-1
                        if d[0]=="1.0(1 - D)":
                            j=0
                        elif d[0]=="2.0(1 - D)":
                            j=1
                        elif d[0]=="3.0(1 - D)":
                            j=2
                        if d[1]==4:
                            i=0
                        elif d[1]==16:
                            i=1
                        if d[2]=='flat':
                            mark='o'
                        elif d[2]=='binary':
                            mark='^'
                        elif d[2]=='quad':
                            mark='s'
                        if i not in self.parallel_dict.keys():
                            self.parallel_dict.update({i:{}})
                        for f in np.arange(0,3,1):
                            types=['o','^','s'] if d[1]==16 else ['o','^']
                            for m in types:
                                if (f,m) not in self.parallel_dict.get(i).keys():
                                    tmp=self.parallel_dict.get(i)
                                    tmp.update({(f,m):[0,0,0]})
                                    self.parallel_dict.update({i:tmp})
                        if dots[i][0][0] is None:
                            dots[i][0]=[str(val[0]),str(val[2]),str(i),str(mark),str(j)]
                        else:
                            dots[i]= np.append(dots[i],[[str(val[0]),str(val[2]),str(i),str(mark),str(j)]],axis=0)
                        if lines[i][0][0] is None:
                            lines[i][0] = [str(val[0]),str(val[2]),str(i),str(mark),str(j)]
                        else:
                            add=True
                            for l in range(len(lines[i])):
                                y_ref=float(val[2])-float(lines[i][l][1])
                                x_ref=float(val[0])-float(lines[i][l][0])
                                if x_ref>=0 and y_ref<=0:
                                    add=False
                            if add:
                                to_erase=[]
                                for l in range(len(lines[i])):
                                    y_ref=float(val[2])-float(lines[i][l][1])
                                    x_ref=float(val[0])-float(lines[i][l][0])
                                    if x_ref<=0 and y_ref>=0:
                                        to_erase.append(l)
                                lines[i] = np.delete(lines[i],to_erase,axis=0)
                                lines[i] = np.append(lines[i],[[str(val[0]),str(val[2]),str(i),str(mark),str(j)]],axis=0)

                max=-1111
                for i in range(len(lines)):
                    x,y=[],[]
                    for l in range(len(lines[i])):
                        x.append(float(lines[i][l][0]))
                        y.append(float(lines[i][l][1]))
                        ax.scatter(float(lines[i][l][0]),float(lines[i][l][1]),s=700,facecolors=self.colors[int(lines[i][l][2])],edgecolors=self.colors[int(lines[i][l][2])],marker=lines[i][l][3],linewidth=1,alpha=float(self.alphas[int(lines[i][l][4])%len(self.alphas)]))
                    ordered_x,ordered_y=[],[]
                    while True:
                        if len(x)>0:
                            xmin=100000000000000
                            ymin=100000000000000
                            ind_flag=0
                            for l in range(len(x)):
                                xm=x[l]
                                ym=y[l]
                                if xm<xmin and ym<ymin :
                                    xmin=xm
                                    ymin=ym
                                    ind_flag=l
                            ordered_x.append(x[ind_flag])
                            ordered_y.append(y[ind_flag])
                            x.pop(ind_flag)
                            y.pop(ind_flag)
                        else:
                            break
                    if ordered_x[-1]>max:
                        max=ordered_x[-1]
                    ax.plot(ordered_x,ordered_y,color=self.colors[i],linewidth=5,linestyle=self.styles[i])
                for i in range(len(dots)):
                    for l in range(len(dots[i])):
                        if float(k) == 0.75:
                            K=0
                        elif float(k) == 0.85:
                            K=1
                        elif float(k) == 1:
                            K=2
                        tmp_d=self.parallel_dict.get(float(dots[i][l][2]))
                        tmp=self.parallel_dict.get(float(dots[i][l][2])).get((float(dots[i][l][4]),dots[i][l][3]))
                        tmp[K]=float(dots[i][l][1])
                        tmp_d.update({(float(dots[i][l][4]),dots[i][l][3]):tmp})
                        self.parallel_dict.update({float(dots[i][l][2]):tmp_d})
                        sem=True
                        for ii in range(len(lines)):
                            for ll in range(len(lines[ii])):
                                sem=False if float(dots[i][l][0])==float(lines[ii][ll][0]) and float(dots[i][l][1])==float(lines[ii][ll][1]) else True
                                if not sem:
                                    break
                            if not sem:
                                break
                        if sem:
                            ax.scatter(float(dots[i][l][0]),float(dots[i][l][1]),s=350,facecolors=self.colors[int(dots[i][l][2])],edgecolors=self.colors[int(dots[i][l][2])],marker=dots[i][l][3],linewidth=1,alpha=float(self.alphas[int(dots[i][l][4])%len(self.alphas)]))
                plt.ylabel('accuracy')
                plt.xlabel('convergence time (decision steps)')
                plt.ylim(-.05, 1.05)
                plt.xlim(0, 420)
                ax.set_yticks(np.arange(0,1.1,.1))
                ax.grid(True,linestyle='--')

                plt.tight_layout()
                plt.savefig(base+'/'+str(k)+"k_pareto_diagram.png")
                print(str(k)+"k_pareto_diagram.png saved in :", base)
                # plt.show()
                plt.close(fig)
            fig, ax = plt.subplots(2,figsize=(8, 9))
            x=0
            flat = mlines.Line2D([], [], color='black', marker='None', markerfacecolor='silver', linestyle=':', markeredgewidth=2.5, markersize=10, label='Flat tree')
            binary = mlines.Line2D([], [], color='black', marker='None', markerfacecolor='silver', linestyle='--', markeredgewidth=2.5, markersize=10, label='Binary tree')
            quad = mlines.Line2D([], [], color='black', marker='None', markerfacecolor='silver', linestyle='-.', markeredgewidth=2.5, markersize=10, label='Quad tree')

            void = mlines.Line2D([], [], linestyle='None')

            r1 = mlines.Line2D([], [], color=self.par_colors[0], marker='_', linestyle='None', markeredgewidth=5, markersize=14, label='r=1-Hd')
            r2 = mlines.Line2D([], [], color=self.par_colors[1], marker='_', linestyle='None', markeredgewidth=5, markersize=14, label='r=2(1-Hd)')
            r3 = mlines.Line2D([], [], color=self.par_colors[2], marker='_', linestyle='None', markeredgewidth=5, markersize=14, label='r=3(1-Hd)')

            handl1 = [flat,r1]
            handl2 = [binary,r2]
            handl3 = [quad,r3]
            ax[0].legend(handles=handl1+handl2+handl3, ncol=3,loc='lower left',bbox_to_anchor=(0,1.1,1,1),mode='expand')
            sorted_keys=np.sort(list(self.parallel_dict.keys()))
            for k in sorted_keys:
                ax[x].xaxis.set_ticks([0,1,2])
                for kk in self.parallel_dict.get(k).keys():
                    if kk[1]=='o':
                        stl=0
                    elif kk[1]=='^':
                        stl=1
                    elif kk[1]=='s':
                        stl=2
                    ax[x].plot([0,1,2],[self.parallel_dict.get(k).get(kk)[0],self.parallel_dict.get(k).get(kk)[1],self.parallel_dict.get(k).get(kk)[2]],color=self.par_colors[int(kk[0])],linewidth=5,linestyle=self.styles[stl])

                ax[x].grid(True,linestyle='--')
                ax[x].set_yticks(np.arange(0,1.1,.1))
                ax[x].set_ylabel('accuracy')
                if x==1:
                    ax[x].set_xlabel(r'$\kappa$')
                    ax[x].xaxis.set_ticklabels([0.75,0.85,1])
                if x==0:
                    ax[x].xaxis.set_ticklabels(['','',''])
                if k==0:
                    num=4
                if k==1:
                    num=16
                ax[x].set_title('N='+str(num))
                x+=1
            plt.ylim(-.05, 1.05)
            plt.tight_layout()
            plt.savefig(base+'/'+"parallel_diagram.png")
            print("parallel_diagram.png saved in :", base)
            # plt.show()
            plt.close(fig)

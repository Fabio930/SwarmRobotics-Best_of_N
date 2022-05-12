# -*- coding: utf-8 -*-
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import numpy as np
import os, csv, time, sys
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import scipy.special as sc

def weib_cdf(x,alpha,gamma):
    return (1-np.exp(-np.power(x/alpha,gamma)))

def weibull_plot(array,figPath,step,max,ref=0):
    col=0
    for i in array:
        if i is not None:
            col+=1
    if col!=0:
        fig, ax = plt.subplots(figsize=(14, 8))
        for i in range(len(array)):
            if array[i] is not None:
                flag=[]
                for j in range(len(array[i][2])):
                    if j==0:
                        flag.append(array[i][2][j])
                    else:
                        while flag[-1]<array[i][2][j]-1:
                            flag.append(flag[-1]+1)
                        flag.append(array[i][2][j])
                y_weib=weib_cdf(flag,array[i][3][0],array[i][3][1])
                ax.plot(flag,y_weib,linewidth=1.5,linestyle='--',label="Weibull Distribution",color=plt.cm.tab10(i) if len(array)<=10 else plt.cm.tab20(i))
                ax.plot(array[i][2],array[i][4],linewidth=2,label="K-M, r="+str(array[i][7]),color=plt.cm.tab10(i) if len(array)<=10 else plt.cm.tab20(i))
        if col>1:
            plt.legend(loc='best',borderaxespad=0,ncol=int(col/2))
        else:
            plt.legend(loc='best',borderaxespad=0)
        ax.set_yticks(np.arange(0,1.1,.05))
        plt.grid(True,linestyle=':')
        plt.ylim(-.05,1.05)
        add=''
        if ref==0:
            ax.set_xticks(np.arange(0,max,step*4))
            add='steps'
        elif ref==1:
            ax.set_xticks(np.arange(0,max,10))
            add='decisions'
        plt.xlabel("Number of "+add)
        plt.ylabel("Synchronisation probability")
        path=''
        for i in range(len(figPath)):
            if figPath[i]=='.' and figPath[i+1]=='c' and figPath[i+2]=='s' and figPath[i+3]=='v':
                path=path+'_'+add+'.png'
                break
            else:
                path+=figPath[i]
        plt.tight_layout()
        plt.savefig(path)
        # plt.show()
        plt.close(fig)

#####################################################################################################
#####################################################################################################

class Plots:

    def __init__(self):
        self.bases=[]
        self.base = os.path.abspath("")
        for elem in os.listdir(self.base):
            if elem[0]=='r' and elem[1]=='e' and elem[2]=='s':
                self.bases.append(os.path.join(self.base, elem))

    def plot_heatmap(self,sub_path,sub_dir,elem):
        fig,ax = plt.subplots(figsize=(11,5))
        plt.tick_params(bottom='on')
        hm = pd.read_csv(os.path.join(sub_path, elem))
        hm['x'] = pd.Categorical(hm['x'])
        hm.head()
        hm = hm.pivot_table('N','x','r',fill_value=0)
        h=[]
        for i in hm.values:
            h.append(list(i))
        n_runs=0
        for i in range(len(h[0])):
            sum = 0
            for j in range(len(h)):
                sum+=h[j][i]
            if sum>n_runs:
                n_runs=sum
        ax=sns.heatmap(hm,vmin=0,vmax=n_runs,cmap='coolwarm')
        ax.set_xticks(np.arange(len(hm.columns)))
        ax.set_yticks(np.arange(len(hm.index)))
        ax.set_xticklabels(hm.columns)
        TICKS=np.array([])
        for i in hm.index:
            if round(i%0.1,2) == 0.0 or round(i%0.1,2) == 0.1:
                TICKS = np.append(TICKS,round(i,2))
            else:
                TICKS = np.append(TICKS,'')
        ax.set_yticklabels(TICKS)
        ax.invert_yaxis()
        ax.set_xlabel('r')
        filename=''
        for i in elem:
            if i !='.':
                filename+=i
            else:
                break
        plt.tight_layout()
        plt.savefig(sub_path+'/'+sub_dir+'_'+filename+'.png')
        # plt.show()
        plt.close(fig)

    def plot_convergence(self,sub_path,elem):
        filename=''
        for i in elem:
            if i !='.':
                filename+=i
            else:
                break
        max_steps = 0
        n_runs = 0
        rt = 0
        n_agents = 0
        R=[]
        wMAX=0
        dwMAX=0
        loc_dict = {}
        dec_dict = {}
        dist_dict = {}

        leafs=[]
        with open(os.path.join(sub_path, elem), newline='') as f:
            reader = csv.reader(f)
            s=0
            for row in reader:
                if s==1:
                    rt=int(row[0])
                    max_steps=int(row[1])
                    stri=''
                    for val in row[2]:
                        if val ==' ' or val =='[' or val==']'or val ==',':
                            try:
                                a=int(stri)
                                leafs.append(a)
                                stri=''
                            except:
                                stri=''
                        else:
                            stri=stri+val
                elif s>1:
                    if int(row[3])>n_agents:
                        n_agents=int(row[3])
                    if row[2] not in R:
                        R.append(row[2])
                    if int(row[1])>n_runs:
                        n_runs=int(row[1])
                    loc_arr=[]
                    stri=''
                    for val in row[4]:
                        if val ==' 'or val =='[' or val ==',' or val==']':
                            try:
                                a=int(stri)
                                loc_arr.append(a)
                                stri=''
                            except:
                                stri=''
                        else:
                            stri=stri+val
                    dec_arr=[]
                    stri=''
                    for val in row[5]:
                        if val ==' ' or val =='[' or val ==',' or val==']':
                            try:
                                a=float(stri)
                                dec_arr.append(a)
                                stri=''
                            except:
                                stri=''
                        else:
                            stri=stri+val
                    dist_arr=[]
                    stri=''
                    for val in row[7]:
                        if val ==' ' or val =='[' or val ==',' or val==']':
                            try:
                                a=float(stri)
                                dist_arr.append(a)
                                stri=''
                            except:
                                stri=''
                        else:
                            stri=stri+val

                    loc_dict.update({(int(row[1]),row[2],int(row[3])):[loc_arr,int(row[6])]})
                    dec_dict.update({(int(row[1]),row[2],int(row[3])):dec_arr})
                    dist_dict.update({(int(row[1]),row[2],int(row[3])):dist_arr})
                s+=1
        n_agents+=1
        R=np.sort(R)
        leafs=np.sort(leafs)
        new_loc_dict={}
        new_dec_dict={}
        new_dist_dict={}
        for i in R:
            new_loc_dict.update({i:{}})
            new_dec_dict.update({i:{}})
            new_dist_dict.update({i:{}})
        for i in new_loc_dict.keys():
            for j in range(1,n_runs+1):
                new_loc_dict.get(i).update({j:[np.resize(np.array([0]),(n_agents,(max_steps//rt)+1)),0]})
                new_dec_dict.get(i).update({j:np.resize(np.array([0]),(n_agents,(max_steps//rt)+1))})
                new_dist_dict.get(i).update({j:np.resize(np.array([0]),(n_agents,(max_steps//rt)+1))})
        for k in loc_dict.keys():
            tmp=loc_dict.get((k[0],k[1],k[2]))
            flag=new_loc_dict.get(k[1]).get(k[0])[0]
            flag[k[2]]=tmp[0]
            new_loc_dict.get(k[1]).update({k[0]:[flag,tmp[1]]})

            tmp=dec_dict.get((k[0],k[1],k[2]))
            flag=new_dec_dict.get(k[1]).get(k[0])
            flag[k[2]]=tmp
            new_dec_dict.get(k[1]).update({k[0]:flag})

            tmp=dist_dict.get((k[0],k[1],k[2]))
            flag=new_dist_dict.get(k[1]).get(k[0])
            flag[k[2]]=tmp
            new_dist_dict.get(k[1]).update({k[0]:flag})

        plot_arr=[None]*len(new_loc_dict.keys())
        dplot_arr=[None]*len(new_loc_dict.keys())
        I=-1
        childs=-1
        depth=-1
        tmp0=''
        pareto_path=''
        k_path=True
        k_sem=False
        for i in range(len(sub_path)):
            if k_path:
                pareto_path+=sub_path[i]
            if depth==0 and sub_path[i]!='_':
                tmp0+=sub_path[i]
            elif childs==0 and sub_path[i]!='D':
                tmp0+=sub_path[i]
            if sub_path[i]=='K':
                k_path=False
                pareto_path=pareto_path[:-1]
                childs=0
            elif sub_path[i]=='D':
                childs=int(tmp0)
                depth=0
                tmp0=''
            elif sub_path[i]=='_' and depth==0:
                depth=int(tmp0)
                tmp0=''
            if depth!=-1 and sub_path[i]=='k':
                tmp0=''
                k_sem=True
            elif k_sem:
                tmp0+=sub_path[i]
                if i == len(sub_path)-1:
                    k_sem=float(tmp0[:-2])
                    tmp0=''

        distance={}
        for i in range(depth+2):
            distance.update({i:{}})
            for j in R:
                distance.get(i).update({j:0})
        percentages_temps={}
        for i in R:
            percentages_temps.update({i:[]})
        for i in new_loc_dict.keys():
            I+=1
            tmp=[len(new_loc_dict.get(i).get(1)[0][0])*rt]*len(new_loc_dict.get(i).keys())
            dtmp=[len(new_loc_dict.get(i).get(1)[0][0])*rt]*len(new_loc_dict.get(i).keys())
            for j in new_loc_dict.get(i).keys():
                test=new_loc_dict.get(i).get(j)[0]
                for k in range(0,len(test[0])):
                    sem=0
                    for z in range(len(test)):
                        cum=0
                        for zk in range(len(test)):
                            if z!=zk and test[z][k]==test[zk][k]:
                                for l in leafs:
                                    if test[z][k]==l:
                                        cum+=1
                                        break
                        if cum>(n_agents-1)*.9:
                            sem=1
                            break
                        else:
                            if z>1:
                                break
                    if sem==1:
                        tmp[j-1]=k*rt
                        dtmp[j-1]=k
                        percentages_temps.update({i:tmp})
                        break
                    if k == len(new_loc_dict.get(i).get(1)[0][0])-1:
                        percentages_temps.update({i:tmp})
            NPtmp=np.array(tmp)
            NPtmp=np.sort(NPtmp,axis=None,kind='stable')
            NPtmp=np.insert(NPtmp,0,0)
            censored=[]
            for j in range(len(NPtmp)):
                if NPtmp[j]==len(new_loc_dict.get(i).get(1)[0][0])*rt:
                    s=0
                    for c in censored:
                        if c != 0:
                            s+=c
                    censored.append(s+1)
                else:
                    censored.append(0)
            ones=0
            for j in censored:
                if j==1:
                    ones+=1
            if ones>.9*(len(NPtmp)-1):
                print('r : '+str(i),',not enough entries for mean and std')
            else:
                for k in range(len(NPtmp)):
                    if NPtmp[k]==len(new_loc_dict.get(i).get(1)[0][0])*rt:
                        NPtmp[k]=len(new_loc_dict.get(i).get(1)[0][0])*rt-rt
                flag=[0]*len(NPtmp)
                for k in range(len(NPtmp)):
                    tmp1=len(NPtmp)-(k+1)
                    if tmp1>0:
                        for z in range(k+1,len(NPtmp)):
                            if NPtmp[z]==NPtmp[k]:
                                tmp1-=1
                            else:
                                flag[k]=tmp1
                                break
                RT = [1]
                for k in range(1,len(flag)):
                    if flag[k]==0:
                        RT.append(RT[-1])
                    else:
                        RT.append(RT[-1]*((flag[k]-1)/(flag[k]+censored[k])))
                FT=[]
                for k in RT:
                    FT.append(1-k)
                lim=10
                a=1
                if NPtmp[-1]>wMAX:
                    wMAX=NPtmp[-1]
                while a>.05:
                    popt_weibull,_= curve_fit(weib_cdf,xdata=NPtmp,ydata=FT,bounds=(0,lim),method='trf')
                    mean = sc.gamma(1+(1./popt_weibull[1]))*popt_weibull[0]
                    std_dev = np.sqrt(popt_weibull[0]**2 * sc.gamma(1+(2./popt_weibull[1])) - mean**2)
                    plot_arr[I]=[mean,std_dev,NPtmp,popt_weibull,FT,'test',ones,i]
                    y_weib=weib_cdf(plot_arr[I][2],plot_arr[I][3][0],plot_arr[I][3][1])
                    err=0
                    for k in range(len(FT)):
                        err+=y_weib[k]-FT[k]
                    err=err/len(FT)
                    a=err
                    lim=lim+10
                print('r : '+str(i),',removed: '+str(ones),',mean: '+str(mean),',std: '+str(std_dev),'\n')
            mean_dec=[len(new_loc_dict.get(i).get(1)[0][0])*rt]*(len(dtmp))
            for k in new_dec_dict.get(i).keys():
                if dtmp[k-1]!=len(new_loc_dict.get(i).get(1)[0][0])*rt:
                    sum=0
                    for z in range(len(new_dec_dict.get(i).get(k))):
                        sum+=new_dec_dict.get(i).get(k)[z][dtmp[k-1]]
                    mean_dec[k-1]=sum/n_agents
                else:
                    mean_dec[k-1]=dtmp[k-1]
            dtmp=mean_dec
            dNPtmp=np.array(dtmp)
            dNPtmp=np.sort(dNPtmp,axis=None,kind='stable')
            dNPtmp=np.insert(dNPtmp,0,0)
            MAX_decis=0
            for d in dNPtmp:
                if d!=len(new_loc_dict.get(i).get(1)[0][0])*rt and d>MAX_decis:
                    MAX_decis=d
            if ones<=9*(len(dNPtmp)-1):
                for k in range(len(dNPtmp)):
                    if dNPtmp[k]==len(new_loc_dict.get(i).get(1)[0][0])*rt:
                        dNPtmp[k]=MAX_decis
                dflag=[0]*len(dNPtmp)
                for k in range(len(dNPtmp)):
                    dtmp1=len(dNPtmp)-(k+1)
                    if dtmp1>0:
                        for z in range(k+1,len(dNPtmp)):
                            if dNPtmp[z]==dNPtmp[k]:
                                dtmp1-=1
                            else:
                                dflag[k]=dtmp1
                                break
                dRT = [1]
                for k in range(1,len(dflag)):
                    if dflag[k]==0:
                        dRT.append(dRT[-1])
                    else:
                        dRT.append(dRT[-1]*((dflag[k]-1)/(dflag[k]+censored[k])))
                dFT=[]
                for k in dRT:
                    dFT.append(1-k)
                lim=10
                a=1
                if dNPtmp[-1]>dwMAX:
                    dwMAX=dNPtmp[-1]
                while a>.05:
                    dpopt_weibull,_= curve_fit(weib_cdf,xdata=dNPtmp,ydata=dFT,bounds=(0,lim),method='trf')
                    dmean = sc.gamma(1+(1./dpopt_weibull[1]))*dpopt_weibull[0]
                    dstd_dev = np.sqrt(dpopt_weibull[0]**2 * sc.gamma(1+(2./dpopt_weibull[1])) - dmean**2)
                    dplot_arr[I]=[dmean,dstd_dev,dNPtmp,dpopt_weibull,dFT,'test',ones,i]
                    dy_weib=weib_cdf(dplot_arr[I][2],dplot_arr[I][3][0],dplot_arr[I][3][1])
                    derr=0
                    for k in range(len(dFT)):
                        derr+=dy_weib[k]-dFT[k]
                    derr=derr/len(dFT)
                    a=derr
                    lim=lim+10
        self.print_data_weib(filename,plot_arr,sub_path,elem,rt,wMAX)
        self.print_data_weib(filename,dplot_arr,sub_path,elem,rt,dwMAX,1)
        for j in R:
            temp_arrivals=percentages_temps.get(j)
            for i in range(len(temp_arrivals)):
                if temp_arrivals[i]==len(new_loc_dict.get(j).get(1)[0][0])*rt:
                    distance.get(depth+1).update({j:distance.get(depth+1).get(j) + 1/n_runs})
                else:
                    for h in range(2,len(new_loc_dict.get(j).get(i+1)[0])-2):
                        if new_loc_dict.get(j).get(i+1)[0][h][temp_arrivals[i]//rt] in leafs and new_loc_dict.get(j).get(i+1)[0][h][temp_arrivals[i]//rt]==new_loc_dict.get(j).get(i+1)[0][h-2][temp_arrivals[i]//rt] and new_loc_dict.get(j).get(i+1)[0][h][temp_arrivals[i]//rt]==new_loc_dict.get(j).get(i+1)[0][h-1][temp_arrivals[i]//rt] and new_loc_dict.get(j).get(i+1)[0][h][temp_arrivals[i]//rt]==new_loc_dict.get(j).get(i+1)[0][h+1][temp_arrivals[i]//rt] and new_loc_dict.get(j).get(i+1)[0][h][temp_arrivals[i]//rt]==new_loc_dict.get(j).get(i+1)[0][h+2][temp_arrivals[i]//rt]:
                            dist_val=new_dist_dict.get(j).get(i+1)[h][temp_arrivals[i]//rt]
                            break
                    if dist_val!=-1:
                        distance.get(dist_val).update({j:distance.get(dist_val).get(j) + 1/n_runs})

        type=''
        if depth==1:
            type='flat'
        else:
            if childs==2:
                type='binary'
            elif childs==4:
                type='quad'
        is_new = True
        if os.path.exists(pareto_path+'resume.csv'):
            is_new=False
        with open(pareto_path+'resume.csv','a') as f:
            fieldnames1 = ['k','r','N','type','mean','std','succes']
            writer = csv.DictWriter(f,fieldnames=fieldnames1)
            if is_new:
                writer.writeheader()
            I=0
            for i in R:
                sum = 0
                for j in distance.keys():
                    if k_sem!=1:
                        if j==0:
                            sum=distance.get(j).get(i)
                    else:
                        if j!=depth+1:
                            sum+=distance.get(j).get(i)
                if dplot_arr[I] is not None:
                    writer.writerow({'k':k_sem,'r':i,'N':len(leafs),'type':type,'mean':round(dplot_arr[I][0],2),'std':round(dplot_arr[I][1],3),'succes':round(sum,2)})
                I+=1

        l=list(distance.keys())
        leafs_plot=[[]]*len(l)
        K=0
        for k in distance.keys():
            tmp=np.array([])
            for j in R:
                tmp=np.append(tmp,round(distance.get(k).get(j),4))
            leafs_plot[K]=list(tmp)
            K+=1
        width=.8/(len(list(distance.keys())))
        barplt=plt.figure(figsize=(14, 8))
        ax = barplt.add_subplot()
        plt.grid(axis='y')
        rects=np.array([])
        q=[]
        for i in range(len(R)):
            q.append(i)
        q=np.sort(q)
        for i in range(len(leafs_plot)):
            rects=np.append(rects,ax.bar(q+(width*i),leafs_plot[i],width))
        ax.set_xticks(q+width)
        ax.set_yticks(np.arange(0,1.01,.05))
        ax.set_xticklabels(R)
        ax.set_xlabel('r')
        ax.set_ylabel('percentage')
        l.pop(-1)
        l.append('no decision')
        ax.legend([rects[len(R)*i] for i in range(len(l))],l,title='distance',loc=3, bbox_to_anchor=(0,1.02,1,0.2),borderaxespad=0,ncol=len(list(distance.keys())))
        plt.tight_layout()
        # plt.show()
        plt.savefig(sub_path+'/percentages.png')
        plt.close(barplt)

    def print_data_weib(self,filename,p_arr,sub_path,elem,RT,wM,ref=0):
        mean_y=[]
        std_e=[]
        r_x=[]
        add=''
        for k in range(len(p_arr)):
            if p_arr[k] is not None:
                mean_y.append(round(p_arr[k][0],2))
                std_e.append(round(p_arr[k][1],2))
                r_x.append(p_arr[k][7])
        fig,ax = plt.subplots(figsize=(10,5))
        ax.errorbar(r_x, mean_y, std_e, linestyle='None',fmt='-', marker='o')
        ax.set_xlabel('r')
        if ref == 0:
            add='steps'
        elif ref==1:
            add='decisions'
        ax.set_ylabel(add)
        plt.grid(True,linestyle=':')
        if ref == 0:
            add='_'+add
        elif ref==1:
            add='_'+add
        plt.tight_layout()
        plt.savefig(sub_path+'/means_std_'+filename+add+'.png')
        # plt.show()
        plt.close()
        weibull_plot(p_arr, os.path.join(sub_path, elem),RT,wM,ref)

    def plot_estimates(self,sub_path,sub_dir,elem):
        n_runs=0
        leafs=np.array([])
        beahv=np.array([])
        rt=0
        dict = {}
        with open(os.path.join(sub_path, elem), newline='') as f:
            reader = csv.reader(f)
            s=0
            for row in reader:
                if s==1:
                    rt=int(row[3])
                if s>1:
                    arr=[]
                    for val in row[1]:
                        if val ==' ' or val =='[' or val =='[' or val==',':
                            try:
                                a=float(stri)
                                arr.append(a)
                                stri=''
                            except:
                                stri=''
                        else:
                            stri=stri+val
                    dict.update({(row[3],float(row[2]),int(row[0])):[arr,int(row[4])]})
                    if int(row[0])>n_runs:
                        n_runs=int(row[0])
                    if float(row[2]) not in leafs:
                        leafs = np.append(leafs,float(row[2]))
                    if row[3] not in beahv:
                        beahv= np.append(beahv,row[3])
                s+=1
        leafs=np.sort(leafs)
        beahv=np.sort(beahv)
        mean = {}
        for i in beahv:
            for j in leafs:
                flag = None
                for k in range(1,n_runs+1):
                    if flag is None:
                        flag = dict.get((i,j,k))[0]
                    else:
                        for z in range(len(flag)):
                            flag[z]+=dict.get((i,j,k))[0][z]
                for z in range(len(flag)):
                    flag[z]=flag[z]/n_runs
                mean.update({(i,j):[flag,dict.get((i,j,1))[1]]})
        MAXy=0
        for j in leafs:
            for i in beahv:
                if np.amax(mean.get((i,j))[0])>MAXy:
                    MAXy=np.amax(mean.get((i,j))[0])
        MAXy+=1
        for j in leafs:
            fig,ax = plt.subplots(figsize=(14,8))
            for i in range(len(beahv)):
                ax.plot(range(len(mean.get((beahv[i],j))[0])),mean.get((beahv[i],j))[0],linewidth=2,label='r: '+str(beahv[i])+' best node id:'+str(mean.get((beahv[i],j))[1]),color=mpl.cm.viridis(i/len(beahv)))
            plt.legend(loc='upper right')
            title=''
            if j==0:
                title='Global error estimate'
            else:
                title='Error estimate of leaf '+str(j)
            ax.set_yticks(np.arange(0,MAXy,.5))
            ax.set_xticks(np.arange(0,len(mean.get((beahv[0],j))[0]) ,rt*.5))
            plt.grid(True,linestyle=':')
            plt.ylabel("Error")
            plt.xlabel("Steps x"+str(rt))
            filename=''
            for i in elem:
                if i !='.':
                    filename+=i
                else:
                    break
            plt.tight_layout()
            # plt.show()
            plt.savefig(sub_path+'/'+sub_dir+'_'+filename+'_'+title+'.png')
            plt.close(fig)

    def print(self):
        for base in self.bases:
            for dir in os.listdir(base):
                path=os.path.join(base, dir)
                if '.' not in path:
                    for sub_dir in os.listdir(path):
                        temp=os.path.join(path, sub_dir)
                        for folder in os.listdir(temp):
                            sub_path=os.path.join(temp,folder)
                            for elem in os.listdir(sub_path):
                                if elem[-1]=='v' and elem[-2]=='s' and elem[-3]=='c' and elem[-4]=='.' and elem[-5]=='n' and elem[-6]=='o' and elem[-7]=='i' and elem[-8]=='t' and elem[-9]=='a' and elem[-10]=='c':
                                    print('\n===========================================================================\n',os.path.join(sub_path, elem),'\n===========================================================================')
                                    self.plot_convergence(sub_path,elem)
                                elif elem[-1]=='v' and elem[-2]=='s' and elem[-3]=='c' and elem[-4]=='.' and elem[-6]=='H' and elem[-5]== 'M':
                                    print('\n===========================================================================\n',os.path.join(sub_path, elem),'\n===========================================================================')
                                    self.plot_heatmap(sub_path,sub_dir,elem)
                                elif elem[-1]=='v' and elem[-2]=='s' and elem[-3]=='c' and elem[-4]=='.' and elem[-5]=='e' and elem[-6]=='t' and elem[-7]=='a' and elem[-8]=='m' and elem[-9]=='i' and elem[-10]=='t':
                                    print('\n===========================================================================\n',os.path.join(sub_path, elem),'\n===========================================================================')
                                    self.plot_estimates(sub_path,sub_dir,elem)

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import gc

import warnings
warnings.filterwarnings('ignore')

class TraceSom(): 
    def __init__(self, shape, elasticity, lrate):
        self._glob_err = 0
        '''
        Calcul de _fen
        source : https://fr.wikipedia.org/wiki/Moyenne_mobile#Moyenne_mobile_exponentielle
        pour une fenêtre de N alpha = 2/(N+1) et moy = alpha * new + (1-alpha) moy  
        => fenêtre de 99 : alpha = .02 
        => fenêtre de 49 : alpha = .04
        '''
        self._fen = .95
        self._dists = []
        self._glob_err = 0
        self._shape = shape
        self._elasticity = elasticity
        self._lrate = lrate;

        # mémorise le nombre de fois ou chaque neurone est vainqueur
        self._win = np.zeros(self._shape[0:2]) 
        
        self._l_eg=[[None]]*shape[0]
        for i in range(shape[0]):
            self._l_eg[i] = [None] * shape[1]
            for j in range(shape[1]):
                self._l_eg[i][j] = [0.]
        self._max_loc  = 0
        self._max_dist = 0
        self._title = ["Erreur globale",
                       "Répartition des neurones vainqueurs",
                       "Répartition 2D des neurones vainqueurs",
                       "Erreurs locales", "Norm erreurs locales"]

                
    def trace_learn(self, distance, winner):
        self._win[winner] += 1
        
        self._glob_err = self._fen * self._glob_err + (1-self._fen) * distance
        self._dists.extend([self._glob_err])
        # print("distance : ", distance, "\r")
        
        if distance > self._max_dist and len(self._l_eg[winner[0]][winner[1]]) > 1 :
            # print("ancienne distance : ", self._max_dist, "   Nlle distance : ", distance)
            self._max_dist = distance
        
        self._l_eg[winner[0]][winner[1]].append(distance)
        if self._max_loc < len(self._l_eg[winner[0]][winner[1]]):
            self._max_loc = len(self._l_eg[winner[0]][winner[1]])

            
            
    def plot_dists(self):
        plt.plot(self._dists)
        # print("err glob :", self._dists)
        
    def plot_winners(self):
        # print("plot_winners")
        pourcent = 100/self._win.sum()
        ax=plt.gca().axes
        # ax.set_ylim(0, 100)
        plt.ylabel('% gagnant')
        # print("self._win.shape : ",self._win.shape)
        # print("self._win.flatten().shape : ",self._win.flatten().shape)
        # print("self._nbNeurons : ", self._nbNeurons)

        # plt.plot(self._win.reshape(self._nbNeurons)*pourcent, 'yo')
        plt.plot(self._win.flatten()*pourcent, 'yo')

    def plot2D_winners(self):
        pourcent = 100/self._win.sum()
        # graph = plt.imshow(self._win*pourcent, cmap=plt.cm.jet)
        graph = plt.imshow(self._win*pourcent, cmap=plt.cm.viridis)
        plt.colorbar(graph) # adding the colobar on the right
        plt.ylabel('% gagnant')

    def plot_norm_local_err(self, row=None, col=None, deb=1):
        '''
        erreur locales normalisées
        '''
        if row is None:
            row=self._shape[0]
        if col is None:
            col=self._shape[1]
        plt.subplots_adjust(wspace=0, hspace=0)
        # print("max local = ", self._max_loc)
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                # plt.subplot(self._shape[0],self._shape[1],deb+i*self._shape[0]+j+1)
                plt.subplot(row, col, deb+i*self._shape[0]+j)
                #plt.axis('off')
                ax=plt.gca().axes
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.set_xlim(1, self._max_loc)
                ax.set_ylim(1, self._max_dist)
                plt.plot(self._l_eg[i][j], '.g', markersize=1)          
        
        
    def plot_local_err(self, row=None, col=None, deb=1):
        if row is None:
            row=self._shape[0]
        if col is None:
            col=self._shape[1]
        plt.subplots_adjust(wspace=0, hspace=0)
        # print("max local = ", self._max_loc)
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                # plt.subplot(self._shape[0],self._shape[1],deb+i*self._shape[0]+j+1)
                plt.subplot(row, col, deb+i*self._shape[0]+j)
                #plt.axis('off')
                ax=plt.gca().axes
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.set_xlim(1, self._max_loc)
                plt.plot(self._l_eg[i][j], '.g', markersize=1)          
        
    def plots(self, ind):
        title = self._title[ind]
        if plt.fignum_exists(title):
            exist=True
        else:
            exist=False
            
        plt.rcParams['toolbar'] = 'None'         
        
        if ind == 0:                # err glob
            plt.figure(num=title, figsize=(2,1))
            plt.clf()
            self.plot_dists()
        elif ind == 1:              # rep winners 1D
            plt.figure(num=title, figsize=(5,2))
            plt.clf()
            self.plot_winners()
        elif ind == 2:              # rep winners 2D
            plt.figure(num=title, figsize=(2,1))
            plt.clf()
            self.plot2D_winners()            
        elif ind == 3:              # erreurs locales
            plt.figure(num=title, figsize=(3,3))
            plt.clf()
            self.plot_local_err()
        elif ind == 4:              # erreurs locales
            plt.figure(num=title, figsize=(3,3))
            plt.clf()
            self.plot_norm_local_err()
        
        if exist :
            plt.figure(title).canvas.draw()
        else:
            plt.ion()
            plt.show()
        plt.pause(0.001)

    def plot_all(self):
        for ind in range(len(self._title)-1):
            # print("ouverture de :", ind, " ", self._title[ind])
            self.plots(ind)

    def close_plots(self):
        '''
        for ind in range(len(self._title)):
            print("fermeture de :", ind, " ", self._title[ind])
            plt.close(ind)
            plt.pause(0.002)
        '''
        plt.close('all')
        gc.collect()
        plt.pause(0.002)
        # print("sortie close_plots()")
            
    def show_plots(self):
        print("Fermer les fenêtres de courbes pour terminer le programme")              
        dir = "/tmp/"
        base="{}/dsom{}x{}_{}e_{}l.".format(dir, self._shape[0], self._shape[1], self._elasticity, self._lrate)
        # debug 17 dec.21 Problème d'ouverture des bilans si ouverture fenêtre pdt expe
        plt.close('all')
        plt.pause(0.001)

        plt.rcParams['toolbar'] = 'toolmanager'

        plt.pause(0.001)
        plt.ioff()

        row, col = 2, 6
        figure = plt.figure(num="résultats", figsize=(10,7))
       
        plt.subplot(row,col,(1, 3))
        self.plot_winners()
        plt.title("Répartition des neurones vainqueurs")
        
        plt.subplot(row,col, (5,6))
        self.plot2D_winners()
        plt.title("Répartition des neurones vainqueurs")

        plt.subplot(row,col,(7, 12))
        self.plot_dists()
        plt.title("Erreur globale")
        plt.savefig('{}glob.png'.format(base), format='png')

        plt.figure(num="erreurs locales", figsize=(10,7))
        self.plot_local_err()
        # plt.title("Erreurs locales")
        plt.savefig('{}loc.png'.format(base), format='png')
        '''#plt.show()
        '''
        plt.figure(num="erreurs locales normalisées", figsize=(10,7))
        self.plot_norm_local_err()
        plt.savefig('{}locnorm.png'.format(base), format='png')
        plt.show()


        # sauver les erreurs globales
        a_file = open("{}err_glob.txt".format(base), "w")
        # print("self._dists.shape()", self._dists.shape())
        # print("len(self._dists) :", len(self._dists))
        for erreur in self._dists:
            # print("erreur : ",erreur)
            a_file.write(str(erreur)+"\n")
        a_file.close()
        

    def show_plots1(self) :
        plt.close('all')
        plt.rcParams['toolbar'] = 'toolmanager'
        plt.pause(0.001)
        plt.ioff()
        fig = plt.figure(num="résultats", figsize=(10,7))
        outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)
        inner = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=outer[1],
                                         wspace=0.1, hspace=0.1)
        ax=plt.subplot(fig, inner[0,1])
        self.plot_winners(ax)
        
        #self.plot2D_winners()
        #self.plot_dists()
        
        plt.show()
        # sauver les erreurs globales
        a_file = open(".txt", "w")
        for row in an_array:            
            np.savetxt(a_file, row)
        a_file.close()

        
        print("[Courbes] Fermer la fenêtre de courbes pour terminer le programme")      

    def help(self):
        help  = "\t < d > \t trace grille 2D neurones vainqueurs \n"
        help += "\t < e > \t trace l'erreur globale \n"
        help += "\t < c > \t trace l'erreur locale \n"
        help += "\t < g > \t trace erreur globale , erreurs locales & vainqueurs \n"
        help += "\t < z > \t ferme les fênêtres de trace \n"
        return help

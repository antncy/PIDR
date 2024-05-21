# -*- coding: utf-8 -*-
### 
###   Interface graphique de som
###   basée sur cv2
### 
import classifieurs.network as net
import cv2
import numpy as np
import math
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

class GuiSom():        
    ''' 
    Fenêtre de suivi d'une SOM
    :Parameters:
    `som` : Classe de réseau disponible dans classifieur.network
    'width' : taille de la fenêtre de rendu du réseau
    'mainFrame' : fenêtre de capture opencv
    '''
    
    @staticmethod
    def raz_neuron(width):
        frame = np.full((width, width, 3), 200, np.uint8)    
        return frame


    def display_neuron(self, neuron):
        vector = self._network.codebook[neuron[0], neuron[1]]
        largeur = self._widthNeuron
        
        positionx = neuron[0] * largeur
        positiony = neuron[1] * largeur
    
        ## Mise à blanc du neurone
        neurframe = self.raz_neuron(self._widthNeuron)
    
        if neuron == self._winner:
            color = (0, 0, 255)
        else:
            color = (0, 0, 0)
    
        cv2.rectangle(neurframe, (0,0), (largeur-1, largeur-1), color)
    
        ## Affichage numéro neuron        
        cv2.putText(neurframe, "neur {}".format(neuron[0]*self._shape[0]+neuron[1]), (3, largeur -3),
                    cv2.FONT_HERSHEY_SIMPLEX, .3, color, 1)
        ## Affichage nb vainqueur neuron       
        cv2.putText(neurframe, "win {}".format(self._win[neuron]), (largeur -30, largeur -3),
                    cv2.FONT_HERSHEY_SIMPLEX, .2, color, 1)
 


        ## Affichage des points de saillance
        i = 0
        while i < len(vector):
            #print("vector[",i,"] = (", vector[i], ",",vector[i+1],")")
            px = int(round(largeur * vector[i]))
            py = int(round(largeur * vector[i+1]))
            cv2.circle(neurframe,(px, py), 1, 100, -1)
            ## if (i>24 and i<36) or i < 14:# or i >20:
            ##     cv2.putText(neurframe, "{}".format(int(i/2)), (px, py), cv2.FONT_HERSHEY_SIMPLEX, .5,
            ##         color, 1)
            i = i+2
        ## remplacer le neurone dans frame
        self._frame[positionx:(positionx + largeur),positiony:(positiony + largeur)] = neurframe
    
    
    def display_map (self):
        # print(" [Som Gui] display_map")
        for i in range(self._shape[0]):
            for j in range (self._shape[1]):
                self.display_neuron((i,j))

                
    def init_minidisp(self):
        largeur = self._width
        # print(" [Som Gui] minidisp, largeur = ",  self._width)
        frame = np.full((largeur, largeur, 3), 250, np.uint8)
        return frame

    
    def track_elast(self, val):
        self._elasticity=float(val/10)
        print("Nouvelle élasticité : ", self._elasticity, "                                          ")

    def track_lrate(self, val):
        self._lrate=float(val/100)
        print("Nouveau taux d'apprentissage : ", self._lrate, "                                          ")

    # ------------------GuiSom::__init__
    def __init__(self, shape, elasticity, lrate, initMethod, classe="DSOM", width=500, name="som"):
        self._width = width
        self._name = name
        self._elasticity = elasticity
        self._lrate = lrate;
 
        self._frame = self.init_minidisp()
        self._shape = shape
        self._nbNeurons = np.delete(np.array(self._shape), [-1]).prod()
        # print("self._nbNeurons :", self._nbNeurons)
        self._widthNeuron = int(math.floor(self._width / shape[0]))
        if  shape[0] *  self._widthNeuron > self._width :
            print("Problème calcul taille du neurone")
            exit()

        self._winner     = None
        self._distance   = None
        '''
        Calcul de _fen
        source : https://fr.wikipedia.org/wiki/Moyenne_mobile#Moyenne_mobile_exponentielle
        pour une fenêtre de N alpha = 2/(N+1) et moy = alpha * new + (1-alpha) moy  
        => fenêtre de 99 : alpha = .02 
        => fenêtre de 49 : alpha = .04
        '''
        ''' 
        3dec21
        self._fen = .95
        self._dists = []
        self._glob_err = 0
        '''

        '''
        listes de liste des erreurs des neurones vainqueurs
        '''
        ''' 
        3dec21
       
        self._l_eg=[[None]]*shape[0]
        for i in range(shape[0]):
            self._l_eg[i] = [None] * shape[1]
            for j in range(shape[1]):
                self._l_eg[i][j] = [0.]
        self._max_loc=0
        self._title = ["Erreur globale",
                 "Répartition des neurones vainqueurs",
                 "Répartition 2D des neurones vainqueurs",
                 "Erreurs locales"]
        '''
        self.init_som(initMethod, classe)
        # creer fenetre som
        self.create_som_view()
        
    def init_som(self, initMethod, classe):
        module = __import__("classifieurs.network", globals(), locals(), classe)        
        classeNetwork = getattr(module, classe)
        print("\t Réseau : {0} : {1}".format(classe, self._shape)) 
        self._network = classeNetwork(self._shape, init_method=initMethod, elasticity=self._elasticity)
        self._win = np.zeros(self._shape[0:2]) # mémorise le nombre de fois ou chaque neurone est vainqueur
        self._last_win = np.zeros(self._shape[0:2]) # mémorise le nombre de fois ou chaque neurone est vainqueur depuis dernière initialisation

        
    def get_and_raz_max_last_win(self):
        last_winner =  np.unravel_index(np.argmax(self._last_win), self._last_win.shape)
        self._last_win[last_winner] = 0
        return last_winner

    def raz_last_win(self):
        self._last_win = np.zeros(self._shape[0:2])
        
    def get_and_raz_last_win(self):
        last_winner_1 =  np.unravel_index(np.argmax(self._last_win), self._last_win.shape)
        self._last_win[last_winner_1] = 0
        last_winner_2 =  np.unravel_index(np.argmax(self._last_win), self._last_win.shape)        
        self._last_win = np.zeros(self._shape[0:2])
        return last_winner_1, last_winner_2
                                  
    def set_win(self):
        return np.unravel_index(np.argmax(self._win), self._win.shape)

    '''
    9 dec. 2021
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
        
        if   ind == 0:                # err glob
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
        
        if exist :
            plt.figure(title).canvas.draw()
        else:
                plt.ion()
                plt.show()
        plt.pause(0.001)

    def plot_all(self):
        for ind in range(len(self._title)):
            self.plots(ind)
            
    def show_plots(self):
        dir = "/tmp/"
        base="{}/dsom{}x{}_{}e_{}l.".format(dir, self._shape[0], self._shape[1], self._elasticity, self._lrate)
        plt.close('all')
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
        plt.show()

        # sauver les erreurs globales
        a_file = open("{}err_glob.txt".format(base), "w")
        # print("self._dists.shape()", self._dists.shape())
        # print("len(self._dists) :", len(self._dists))
        for erreur in self._dists:
            # print("erreur : ",erreur)
            a_file.write(str(erreur)+"\n")
        a_file.close()
        
        print("Fermer la fenêtre de courbes pour terminer le programme")      

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
    '''
        
    def learn(self, data, coefs=None):
        sigma = 0
        # print(" [Som Gui] learn")
        self._winner, self._distance = self._network.learn_data(data, self._lrate, sigma, self._elasticity, coefs=None)

        # print(" [Som Gui] learn... winner = ",  self._winner)        
        self._win[self._winner] += 1
        self._last_win[self._winner] += 1
        # print(" [Som Gui] display_map()")
        self.display_map()

        '''
        9 dec 21
        self._glob_err = self._fen * self._glob_err + (1-self._fen) * self._distance
        self._dists.extend([self._glob_err])

        self._l_eg[self._winner[0]][self._winner[1]].append(self._distance)
        if self._max_loc < len(self._l_eg[self._winner[0]][self._winner[1]]):
            self._max_loc = len(self._l_eg[self._winner[0]][self._winner[1]])
        '''
        
    def print_win(self):
        print("Répartition des vainqueurs")
        print(self._win)
    
    def print_last_win(self):
        print("Répartition des derniers vainqueurs")
        print(self._last_win)
    

    # ------------------GuiSom::network
    @property
    def network(self):
        return self._network

    @property
    def distance(self):
        if self._distance:
            return self._distance
        else:
             return -1

    @property
    def winner(self):
        if self._winner:
            return self._winner 
        else:
             return -1

    @property
    def win(self):
        return self._win
    @property
    def frame(self):
        return self._frame

    def create_som_view(self):
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("élasticité (x10)", self._name, int(10 *self._elasticity), 20, self.track_elast)
        cv2.createTrackbar("Tx App    (x100)", self._name , int(100*self._lrate), 20, self.track_lrate)

    def show_som_view(self):        
        cv2.imshow(self._name, self._frame)

    def destroy_som_view(self):
        cv2.destroyWindow(self._name)



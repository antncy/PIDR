# -*- coding: utf-8 -*-

"""
programme principal
"""
import argparse
import pickle as pk
import time as t

import cv2
import numpy as np
import math

import imutils
import classifieurs.network as net
import som_gui as sg
import trace_som as ts


from caracteristique.caracteristique import caracteristique
#from clasifieur.network import DSOM_MODEL
from draw.drawer import drawer
from landmark.landmarks_corps import landmarks_corps as landmarks

import sys
from PyQt5.Qt import *


elasticity = 0
lrate = 0
interest = ["left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "bottom_lip", "top_lip", "nose_tip", "nose_bridge"]
win = None



# recuperation des parametres du programme
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=False, default=None, type=str,
                help="chemin de la video")
ap.add_argument("-c", "--camera", required=False, default=0, type=int,
                help="numero de la webcam")
ap.add_argument("-f", "--file", required=False, default="/tmp/data.plt", type=str,
                help="chemin dans lequel on sauvgarde le fichier de plotting")
ap.add_argument("-e", "--elasticity", required=False, default=.5, type=float,
                help="elasticity de la DSOM")
ap.add_argument("-im", "--initial-method", required=False, default='regular', type=str,
                help="methode d'initialisation de la DSOM (regular, fixed or random)")
ap.add_argument("-lr", "--learning-rate", required=False, default=0.05, type=float,
                help="le taux d'apprentissage de la DSOM")
ap.add_argument("-sig", "--sigma", required=False, default=0.5, type=float,
                help="le parametre sigma de la DSOM")
ap.add_argument("-dt", "--delta", required=False, default=10, type=int,
                help="intervale du temps entres les images prises pour les traitements")
ap.add_argument("-s", "--speed", required=False, default=30, type=int,
                help="vitesse du plotting (1 = la vitesse maximale)")
ap.add_argument("-r", "--range", required=False, default=20, type=int,
                help="taille de la plage de plotting")
ap.add_argument("-pca", "--pca-samples", required=False, default=300, type=int,
                help="taille de la plage de l'ACP pour le plotting")
ap.add_argument("-a", "--coef", required=False, default=300, type=int,
                 help="coeficient de normalisation des vecteurs VCC")
ap.add_argument("-n", "--order-n", required=False, default=6, type=int,
                help="ordre de la map. ex: n = 3 implique que le nombre de neurones n*n = 3*3 = 9")
ap.add_argument("-d", "--display", required=False, default="000", type=str,
                help="les figures de plotting a afficher \n c'est une chaine de trois bits XXX où chaque chiffre "
                     "correspond à une figure (1 pour afficher la figure et 0 pour ne pas l'afficher)")
ap.add_argument("-sal", "--salient", required=False, default="1111", type=str,
                help="les points de saillance à prendre en compte \n c'est une chaine de quatre bits XXXX où chaque bit "
                     "correspond à une partie du visage dans l'ordre yeux sourcils lèvres nez")
ap.add_argument("-w", "--window", required=False, default=400, type=int,
                help="Taille de la fenêtre. 400px par défaut")
ap.add_argument("-tr", "--trace", required=False, default=False, action="store_true",
                help="Affiche puis enregistre les erreurs dans /tmp")

args = vars(ap.parse_args())





def send_ploting_data(codebook, vect, FCount, dist, pause=False):
    """
    envoyer les donnees au processus du plotting
    :param net: codebook
    :param vect: la nouvelle donnee "/tmp/data.plt"
    :param FCount: nombre des features
    """
    mat = {"data": np.concatenate((codebook.reshape((-1, FCount)), np.reshape(vect, (-1, FCount)))),
           "target": winner,
           "dist": dist,
           "pause": pause}
    with open(f, "wb") as plot_data:
        pk.dump(mat, plot_data)

def send_ploting_data(codebook, vect, FCount, dist, pause=False):
    """
    envoyer les donnees au processus du plotting
    :param net: codebook
    :param vect: la nouvelle donnee "/tmp/data.plt"
    :param FCount: nombre des features
    """
    mat = {"data": np.concatenate((codebook.reshape((-1, FCount)), np.reshape(vect, (-1, FCount)))),
           "target": winner,
           "dist": dist,
           "pause": pause}
    with open(f, "wb") as plot_data:
        pk.dump(mat, plot_data)


if __name__ == '__main__':

    salient =  args["salient"]
    #nbFeatures = 110-24 # taille des vecteurs (sourcils + yeux + nez + levres) (-24 : lèvres divisées par deux)
    
    #interest, nbFeatures = define_vector(salient)

    # print("[INFO] chargement du predicteur des points de saillances...")
    N = args["order_n"] # order of net matrix
    FCount = N*N  # number of features
    lmk   = landmarks(salient, reduct=False)
    f     = args["file"]
    trace = args["trace"]
    dr    = drawer.fromFile(f, _n=(N*N), _f=FCount, n_first=args["range"], _speed=args["speed"], 
                            _disp=args["display"][:3], pca_samples=args["pca_samples"])
    tailleImage = args["window"]
    nom = "PsyPhiNE"
    cv2.namedWindow(nom)

    shape = (N, N, lmk.sizeData)

    elasticity    =    args["elasticity"]
    lrate         = args["learning_rate"]
    sigma         =         args["sigma"]    
    
    interface = sg.GuiSom(shape, elasticity, lrate, args["initial_method"],
                          "DSOM", width=550)
    # Trace des résultats
    if trace:
        traces = ts.TraceSom(shape, elasticity, lrate)

    print(" paramêtres vidéocapture : ", (args["video"], args["camera"])[args["video"] is None])
    vs = cv2.VideoCapture((args["video"], args["camera"])[args["video"] is None])

    if not vs.isOpened():
        print("[ERROR] impossible de demarrer le flux")
        exit(1)

    # fps = vs.get(cv2.CAP_PROP_FPS)

    print("[INFO] En cours d'execution...")
    print("[INFO] Touche < h > pour la liste des commandes.")
    
    vect = np.zeros((1, FCount))
    vcc = np.zeros((1, FCount))
    i = 0  # compteur de frame
    started = False

    # variable pour éviter le frame par frame dans la lecture d'une vidéo.
    # décalage de deux secondes
    decalage = 0  # int(0 * vs.get(cv2.CAP_PROP_FPS))
    # print(decalage)
    
    while vs.isOpened():
        start = int(round(t.time() * 1000))

        if args["video"] is not None:
            for i in range(decalage):
                vs.read()
         # recuperation d'une image du flux video, la redimensionner pour avoir une largeur de 400 pixels
        # ensuite la convertir l'image en grayscale et y appliquer une egalisation d'histogramme.
        playing, frame = vs.read()

        if not playing:
            break

        frame = imutils.resize(frame, width=tailleImage)
        couleur = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # extraction des points de saillances
        lmk.extract_landmarks(couleur)
            
        # Apprentissage des traits du visage
        if lmk.ready:
            i = i + 1
            interface.learn(lmk.current, coefs=lmk.coefs)
            if trace:
                traces.trace_learn(interface.distance, interface.winner)
            # print(" [main] distance au vainqueur : ", interface.distance, end="\r")
            winner = interface.winner
            # print("winner : ", winner)
            frame = lmk.insert_salient(frame, winner)

            if not started:
                dr.start()
                started = True

        # dessiner le numero de frame
        end = (int(round(t.time() * 1000)) - start)
        cv2.putText(frame, "Process Time : {:.2f} ms".format(end), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 1)
        # affichage de l'image
        cv2.imshow(nom, frame)
        


        #Mise à jour fenêtre neurones
        #interface.update()
        interface.show_som_view()
        
        # Attendre la touche 'q' pour sortir
        # ou la touche 'p' pour suspendre le programme

        help  = "AIDE : \t < w > \t affiche les neurones vainqueurs \n"
        help += "\t < l > \t affiche les derniers neurones vainqueurs \n"
        help += "\t < v > \t trace les neurones vainqueurs \n"
        if trace:
            help += traces.help()
            '''
            help += "\t < d > \t trace grille 2D neurones vainqueurs \n"
            help += "\t < e > \t trace l'erreur globale \n"
            help += "\t < c > \t trace l'erreur locale \n"
            help += "\t < g > \t trace erreur globale , erreurs locales & vainqueurs \n"
            '''
        help += "\t < h > \t affiche cette aide \n"
        help += "\t < q > \t fermer le programme \n"


        
        key = cv2.waitKey(1)
        car = chr(key & 255)
        if key == 112 or car=='p':
            while True:
                key = cv2.waitKey(1)
                car = chr(key & 255)
                if  key == 112 or car=='p':
                    break

        if car=="h": # Affichage commandes
            print(help)

        if car=="l": # Affichage des derniers neurones vainqueurs
            interface.print_last_win()

        if car=="w": # Affichage répartition des vainqueurs
            interface.print_win()

        if car=="d": # Affichage répartition 2D des vainqueurs
             if trace:
                traces.plots(2)
            #interface.plots(2)

        if car=="v": # Affichage graphique répartition des vainqueurs
            if trace:
                traces.plots(1)
            # interface.plots(1)
            
        if car=="e": # Affichage graphique erreur globale
            if trace:
                traces.plots(0)
                # interface.plots(0)
            
        if car=="c": # Affichage graphique erreur locale
            if trace:
                traces.plots(3)
            #interface.plots(3)


        if car=="g": #Affichage des courbes
             if trace:
                traces.plot_all()
            #interface.plot_all();
        if key == 113 or car=='q':
            break

    print("[INFO] Sortir du programme...")

    # killing the drawer process
    print("[INFO] terminaison du processus de plotting...")
    #if args["video"] is not None:
        # 9 dec.202
        # interface.show_plots()
    if trace:
        traces.show_plots()
    dr.join()

    # cleaning up
    cv2.destroyAllWindows()
    vs.release()
    interface.print_win()


    print("Au revoir....")

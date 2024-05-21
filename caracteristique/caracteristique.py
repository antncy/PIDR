import math
import numpy as np
from numpy.linalg import norm as dist
from distances.distance import distance
from normalisation.normalisation import normalisation


class caracteristique:

    def __init__(self):
        self.pos_precd = None

    def distence(self, vecteur, img_size):
        surface = img_size[0] * img_size[1]
        facepos = vecteur['facepos']
        facesurface = abs(facepos[0][2] - facepos[0][0]) * abs(facepos[0][1] - facepos[0][3])
        n = normalisation()
        return n.val01(surface / (float(facesurface) * 5))

    def mov(self, vecteur):
        facepos = vecteur['facepos']
        dis = 0
        difx = 0
        dify = 0
        pos = (facepos[0][0] + (facepos[0][2] - facepos[0][0]) / 2
               , facepos[0][0] + (facepos[0][1] - facepos[0][3]) / 2)
        if self.pos_precd is not None:
            difx = (self.pos_precd[0] - pos[0])
            dify = (self.pos_precd[1] - pos[1])
            dis = math.sqrt((difx * difx) + (dify * dify))

        self.pos_precd = pos
        n = normalisation()
        return n.val01(difx), n.val01(dify), n.val01(dis / 10)

    def overture_bouche(self, vecteur):
        d = distence()
        eyel = vecteur['left_eye']
        eyer = vecteur['right_eye']
        facetl = vecteur['top_lip']
        facebl = vecteur['bottom_lip']
        disv = d.cartesienne(facebl[3], facetl[3])
        disv2 = d.cartesienne(facebl[9], facetl[9])
        dis = (disv * disv2 + 0.000001) / (d.cartesienne(eyel[3], eyer[0]) * 100 + 0.000001)  # (dish+1)
        n = normalisation()
        return n.val01(dis)

    def sourcils(self, vector):
        """
        extraire la distance entre les points de sourcils et le les yeux
        :param vector: points de saillances
        :return: tuple des distances (gauche, groite)
        """

        # recuperer les points des yeux
        l_eye = vector["left_eye"]
        r_eye = vector["right_eye"]

        # recuperer les points des sourcils
        l_brow = vector["left_eyebrow"]
        r_brow = vector["right_eyebrow"]

        return (dist(l_brow[2] - l_eye.mean(0)) / dist(l_brow[-1] - l_brow[0]),
                dist(r_brow[2] - r_eye.mean(0)) / dist(r_brow[-1] - r_brow[0]))

    def h_rotation(self, vector, thd=0.20):
        """
        extraire le taux de rotation du visage
        0.0 -> pas de rotation dans ce sens
        1.0 -> retation maximale dans ce sens
        :param thd: le seuil de sensibilite de rotation
        :param vector:  points de saillances
        :return: couple de valeurs de rotation (gauche, droite)
        """

        # calculer les distances entre le cote droite/gauche du menton et le nez
        # on prend plusieurs points pour stabiliser la distance
        nose = vector["nose_bridge"].mean(0)
        rt = dist(vector["chin"][13:17].mean(0) - nose)
        lt = dist(vector["chin"][0:4].mean(0) - nose)

        # calculer le precentage de rotation
        l_rot = (0.0, 1 - (lt / rt))[rt > lt]
        r_rot = (0.0, 1 - (rt / lt))[lt > rt]

        # on ignore les valeurs < seuil
        return ((0.0, l_rot)[l_rot > thd],
                (0.0, r_rot)[r_rot > thd])

    def eyes(self, vector, thd=0.22):
        """
        extraire le taux d'ouverture des yeux
        0.0 -> yeux fermes
        1.0 -> yeux ouverts au max.
        :param thd: seuil d'ouverture des yeux
        :param vector: points de saillances
        :return: tuple d'ouveture des yeux (gauche, droite)
        """

        # recuperation des points des yeux
        lt = vector["right_eye"]
        rt = vector["left_eye"]

        # calcule d'ouverture
        l_eye = dist(lt[4:6].mean(0) - lt[1:3].mean(0)) / dist(lt[3] - lt[0])
        r_eye = dist(rt[4:6].mean(0) - rt[1:3].mean(0)) / dist(rt[3] - rt[0])

        # on ignore les valeurs < seuil
        return ((0.0, l_eye)[l_eye > thd],
                (0.0, r_eye)[r_eye > thd])

    def extract_features(self, vect, imgsize):
        """
        recuperer tous les caracteristiques du visage
        :return: (dictionnaire et vecteur ) des caracteristiques
        """
        leye, reye = self.eyes(vect)
        lrot, rrot = self.h_rotation(vect)
        lbrow, rbrow = self.sourcils(vect)
        mouth = self.overture_bouche(vect)
        di = self.distence(vect, imgsize)

        return {"eyes": (leye, reye),
                "rotation": (lrot, rrot),
                "eyebrows": (lbrow, rbrow),
                "mouth": mouth,
                "distance": di,
                "position": vect["position"]}, [leye, reye, lrot, rrot, lbrow, rbrow, mouth, di, vect["position"]]
        # "move": self.mov((rect.center().x, rect.center().y))}

    @staticmethod
    def print_features(dict):
        """
        affichage sophistique du dictionnaire des caracteristiques
        :param dict: le dictionnaire
        """
        print("[INFO] FEATURES LIST")
        for key, value in dict.items():
            if key in ["eyes", "rotation", "eyebrows"]:
                left, right = value
                print("\t{} is {:.2f}".format(key + " left", left))
                print("\t{} is {:.2f}".format(key + " right", right))
            elif key == "position":
                print("\t{} is {}".format(key, value))
            else:
                print("\t{} is {:.2f}".format(key, value))

    @staticmethod
    def calculate_vcc(f1, f2, h=400, w=300):
        diag = np.sqrt(h**2 + w**2)
        return np.append(np.subtract(f2[:-1], f1[:-1]), dist(np.reshape(f2[-1], -1) - np.reshape(f1[-1], -1)) / diag).astype(np.float64)

import dlib
import numpy as np
from imutils import face_utils
from numpy import concatenate as cat
import cv2


class landmarks:
    COLORS = {'right_eyebrow': [255, 255, 0],
              'right_eye': [0, 0, 255],
              'left_eyebrow': [255, 255, 0],
              'left_eye': [0, 0, 255],
              'nose_bridge': [0, 255, 255],
              'nose_tip': [0, 255, 255],
              'top_lip': [0, 0, 128],
              'bottom_lip': [0, 0, 128],
              'chin': [255, 0, 0],
              "position": [0, 0, 0]}

    DEFAULT_PREDICTOR = "shape_predictor_68_face_landmarks.dat"

    def __init__(self, salient="1111", predictor=None, reduct=False):
        """
        constructeur
        :param predictor: le chemin complet vers le model entraine
        :param reduct : réduire les traits aux points saillants
        """
        self._reduct = reduct
        # self._reduct = True
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor((self.DEFAULT_PREDICTOR, predictor)[predictor is not None])
        self.define_vector(salient)
        self._ready = False
        


    def init_salient(self, salient):
         self.define_vector(salient)

    def extract_landmarks(self, img):
        """
        extraire les points de saillances du visage
        :param img l'image en greyscale
        :return un couple contenant le dictionnaire des composantes du visage et son centre
         """

        # recuperation de tous les visages sur l'image
        rects = self.detector(img, 0)
        # print(" [LMK] essai")

        # on sort si aucun visage DETECTE
        if rects.__len__() < 1:
            # print(" [LMK] pas de visage")
            self._ready = False
            self._faces = None 
            self._face = None
            self._rect = None
            return

        #print(" [LMK] visage")
        def target_face():
            maxRect = dlib.rectangle(0, 0, 0, 0)
            for rct in rects:
                if rct.area() > maxRect.area():
                    maxRect = rct
            return maxRect

        # detect faces in the grayscale frame)
        rect = target_face()

        # extraction des points de saillances
        points = face_utils.shape_to_np(self.predictor(img, rect))
        # for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        #     print(name,"[",i,",",j,"]")
        # print("points[0:3].len :", len(points[0:3]))
        # print("points[0:3] :", points[0:3])
        # print("points :", points)
        # quit();

        def points_dict():
            if self._reduct:
                return {
                    "chin": points[0:17],
                    "left_eyebrow":  cat(([points[17]],[points[19]],[points[21]])),
                    "right_eyebrow": cat(([points[22]],[points[24]],[points[26]])),
                    "nose_bridge":   cat(([points[27]],[points[30]])),
                    "nose_tip":      cat(([points[31]],[points[33]], [points[35]])),
                    "left_eye":      cat(([points[36]],[points[38]], [points[39]],[points[41]])),
                    "right_eye":     cat(([points[42]],[points[43]], [points[45]],[points[46]])),
                    "top_lip":       cat(([points[48]], [points[51]], [points[54]])),
                    "bottom_lip":    cat(([points[56]], [points[58]])),
                    "facepos": [(int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom()))],
                    "position": [[rect.center().x, rect.center().y]]
                }
            else:
                return {
                    "chin": points[0:17],
                    "left_eyebrow": points[17:22],
                    "right_eyebrow": points[22:27],
                    "nose_bridge": points[27:31],
                    "nose_tip": points[31:36],
                    "left_eye": points[36:42],
                    "right_eye": points[42:48],
                    "top_lip": cat((points[48:55], [points[64]], [points[63]], 
                                    [points[62]], [points[61]], [points[60]])),
                    "bottom_lip": cat((points[54:60], [points[48]], [points[60]], 
                                       [points[67]], [points[66]], [points[65]], [points[64]])),
                    "facepos": [(int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom()))],
                    "position": [[rect.center().x, rect.center().y]]
                }
               
        self._ready = True               # boolean si visage existe
        self._faces = rects              # Tous les visages détectés
        self._face = points_dict()       # dictionnaire des points du visage le plus grand
        # print("face : ", self._face)
        self._rect = rect                # Contour du visage le plus grand
        self.extract_vector();



    def define_vector(self, salient):
        interest = []
        coefs = []
        nbFeatures = 0
        if salient[1] == '1':
            add = ["left_eyebrow", "right_eyebrow"]
            interest.extend(add)
            if self._reduct:
                nbFeatures += 12
                coef = 12 * [12]
            else:
                nbFeatures += 20
                coef = 20 * [20]
            coefs.extend(coef)
        if salient[3] == '1':
            add = ["nose_tip", "nose_bridge"]
            interest.extend(add)
            if self._reduct:
                nbFeatures += 10
                coef = 10 * [10]
            else:
                nbFeatures += 18
                coef = 18 * [18]
            coefs.extend(coef)            
        if salient[0] == '1':
            add = ["left_eye", "right_eye"]
            interest.extend(add)
            if self._reduct:
                nbFeatures += 16
                coef = 16 * [16]
            else:
                nbFeatures += 24
                coef = 24 * [24]
            coefs.extend(coef)
        if salient[2] == '1':
            add = ["bottom_lip", "top_lip"]
            interest.extend(add)
            if self._reduct:
                nbFeatures += 10 
                coef = 10 * [10]
            else:
                nbFeatures += 24 #48
                coef = 24 * [24]
            coefs.extend(coef)
        self._interest = interest
        self._sizeData  = nbFeatures
        self._coefs = np.array(coefs) 
        print("\t profils : ", self._interest)
        print("\t nombre de points : ", self.sizeData)

    def extract_vector(self):
        ## Normaliser les points... 
        # Point de référence du visage 
        refx  = self._face["facepos"][0][0]
        refy  = self._face["facepos"][0][1]
        largx = self._face["facepos"][0][2] - refx
        largy = self._face["facepos"][0][3] - refy

        # print(face)
        result = np.array([])
    
        for k, pt in self._face.items():
            if k == "facepos":
                continue        
            #if "lip" in k: 4 juin 2021 => reduct
            if "lip" in k:
                points = pt[0:6]
                #points = pt[6:12]

                # lire tableau et le reverser dans result
                for (x, y) in points:
                    if k in self._interest:
                        # print("k :",k)
                        px = (x-refx)/largx
                        py = (y-refy)/largy
                        result = np.append(result, px)
                        result = np.append(result, py)
                continue
            for (x, y) in pt:
                if k in self._interest:
                    # print("k :",k)
                    px = (x-refx)/largx
                    py = (y-refy)/largy
                    result = np.append(result, px)
                    result = np.append(result, py)
        # print("len(result) : ", len(result))
        # print()
        # print()
        self._current = result          # Points d'intérêts

    def insert_capture(self, frame):
        """
        Intégrer l'image des points pris en compte dans l'image
        :param frame: l'image
        :param vector : les saillances apprises
        :return: l'image modifie
        """
        vector = self._current
        coin_gauche = (20, 0)
        largeur = 100
        neurframe = np.full((largeur, largeur, 3), 200, np.uint8)    

        cv2.rectangle(neurframe, (0,0), (largeur-1, largeur-1),  (0, 0, 0))
        i = 0
        while i < len(vector):
            px = int(round(largeur * vector[i]))
            py = int(round(largeur * vector[i+1]))
            cv2.circle(neurframe, (px, py), 1, 100, -1)
            i = i+2
        frame[coin_gauche[0]:coin_gauche[0]+largeur,coin_gauche[1]:coin_gauche[1]+largeur] = neurframe
    
        return frame


    def insert_salient(self, frame, winner):
        """
        dessiner les points de saillances, le cadre du visage et le cluster
        :param frame: l'image
        :param face: les points de saillances
        :param cluster: le resultat de clustering
        :return: l'image modifie
        """
        face = self._face
        refx  = face["facepos"][0][0]
        refy  = face["facepos"][0][1]
        largx =  face["facepos"][0][2] - face["facepos"][0][0]
        largy =  face["facepos"][0][3] - face["facepos"][0][1]



        for k, pt in face.items():
            if k == "facepos":
                # rectangle pour entourer le visage
                [(x1, y1, x2, y2)] = pt
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))
                cv2.rectangle(frame, (x1, y1), (x1 + 100, y1 - 18), (0, 0, 255), -1)
                cv2.putText(frame, "Neuron({0},{1})".format(winner[0], winner[1]), 
                            (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                continue
            for (x, y) in pt:
                cv2.circle(frame, (x, y), 1, landmarks.COLORS[k], -1)
                # if k in self._interest:
                #     px = int(round(100 * (x-refx)/largx))
                #     py = int(round(100 * (y-refy)/largy))
                #     cv2.circle(frame, (px, py), 1, landmarks.COLORS[k], -1)                
        frame = self.insert_capture(frame)
        return frame


    @property
    def interest(self):
        return self._interest

    @property
    def sizeData(self):
        return self._sizeData

    @property
    def ready(self):
        return self._ready

    @property
    def current(self):
        return self._current

    @property
    def rect(self):
        if self._ready:
            return self._rect
    @property
    def coefs(self):
        return self._coefs

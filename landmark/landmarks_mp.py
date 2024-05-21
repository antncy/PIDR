import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from imutils import face_utils
from numpy import concatenate as cat

class landmarks_mp:
    COLORS = {'right_eyebrow': [255, 255, 0],
              'right_eye': [0, 0, 255],
              'left_eyebrow': [255, 255, 0],
              'left_eye': [0, 0, 255],
              'nose_bridge': [0, 255, 255],
              'nose_tip': [0, 255, 255],
              'top_lip': [0, 0, 128],
              'bottom_lip': [0, 0, 128],
              'chin': [255, 0, 0]}


    def __init__(self, salient="1111", reduct=False):
        """
        constructeur
        :param predictor: le chemin complet vers le model entraine
        :param reduct : réduire les traits aux face_landmarks.landmark saillants
        """
        self._reduct = reduct
        # self._reduct = True
        self.detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5)
        self.define_vector(salient)
        self._ready = False

    def init_salient(self, salient):
        self.define_vector(salient)

    def extract_landmarks(self, img):
        """
        extraire les face_landmarks.landmark de saillances du visage
        :param img l'image en greyscale
        :return un couple contenant le dictionnaire des composantes du visage et son centre
         """

        # recuperation de tous les visages sur l'image
        rects = self.face_mesh.process(img)

        haut,large = img.shape[0],img.shape[1]
        results = self.detector.process(img)
        # on sort si aucun visage DETECTE
        if not results.detections:

            self._ready = False
            self._faces = None
            self._face = None
            self._rect = None
            return
        else:
            # extraction des face_landmarks.landmark de saillances
            if(rects.multi_face_landmarks != None):
                face_landmarks = rects.multi_face_landmarks[0]
            else: return

        def point_dict(large,haut,bool=True):
            if bool:
                liste = [(int(results.detections[0].location_data.relative_bounding_box.xmin * img.shape[1]),
                             int(results.detections[0].location_data.relative_bounding_box.ymin * img.shape[0]),
                             int(results.detections[0].location_data.relative_bounding_box.width * img.shape[1]) + int(
                                 results.detections[0].location_data.relative_bounding_box.xmin * img.shape[1]),
                             int(results.detections[0].location_data.relative_bounding_box.height * img.shape[0]) + int(
                                 results.detections[0].location_data.relative_bounding_box.ymin * img.shape[0]))]

            return {
                "chin": cat(([face_landmarks.landmark[127]],[face_landmarks.landmark[234]],[face_landmarks.landmark[93]],[face_landmarks.landmark[132]],[face_landmarks.landmark[58]],[face_landmarks.landmark[172]],[face_landmarks.landmark[136]],
                             [face_landmarks.landmark[150]],[face_landmarks.landmark[176]],[face_landmarks.landmark[148]],[face_landmarks.landmark[152]],
                             [face_landmarks.landmark[400]],[face_landmarks.landmark[378]],[face_landmarks.landmark[379]],
                            [face_landmarks.landmark[365]],[face_landmarks.landmark[397]],[face_landmarks.landmark[288]],[face_landmarks.landmark[361]],[face_landmarks.landmark[323]],[face_landmarks.landmark[454]],[face_landmarks.landmark[356]])),

                "left_eyebrow": cat(([face_landmarks.landmark[124]],[face_landmarks.landmark[46]],[face_landmarks.landmark[53]],[face_landmarks.landmark[52]],[face_landmarks.landmark[65]],[face_landmarks.landmark[55]])),
                "right_eyebrow": cat(([face_landmarks.landmark[353]],[face_landmarks.landmark[276]],[face_landmarks.landmark[283]],[face_landmarks.landmark[282]],[face_landmarks.landmark[295]],[face_landmarks.landmark[285]])),
                "nose_bridge": cat(([face_landmarks.landmark[6]], [face_landmarks.landmark[197]], [face_landmarks.landmark[195]], [face_landmarks.landmark[5]],
                                    [face_landmarks.landmark[4]],[face_landmarks.landmark[1]])),
                "nose_tip": cat(([face_landmarks.landmark[102]], [face_landmarks.landmark[64]], [face_landmarks.landmark[240]],[face_landmarks.landmark[97]],
                                 [face_landmarks.landmark[2]],[face_landmarks.landmark[326]])),
                "left_eye": cat(([face_landmarks.landmark[7]], [face_landmarks.landmark[33]], [face_landmarks.landmark[133]], [face_landmarks.landmark[144]],
                                 [face_landmarks.landmark[145]], face_landmarks.landmark[153:155], face_landmarks.landmark[157:161], [face_landmarks.landmark[163]], [face_landmarks.landmark[173]], [face_landmarks.landmark[246]])),
                "right_eye": cat(([face_landmarks.landmark[249]], [face_landmarks.landmark[263]], [face_landmarks.landmark[362]],
                                  [face_landmarks.landmark[373]], [face_landmarks.landmark[374]], face_landmarks.landmark[380:382], face_landmarks.landmark[384:388], [face_landmarks.landmark[390]], [face_landmarks.landmark[398]], [face_landmarks.landmark[466]])),
                "top_lip": cat(( [face_landmarks.landmark[291]], [face_landmarks.landmark[409]], [face_landmarks.landmark[270]],[face_landmarks.landmark[269]],[face_landmarks.landmark[267]], [face_landmarks.landmark[0]], [face_landmarks.landmark[37]], [face_landmarks.landmark[39]], [face_landmarks.landmark[40]],[face_landmarks.landmark[185]], [face_landmarks.landmark[61]])),
                "bottom_lip": cat(([face_landmarks.landmark[61]], [face_landmarks.landmark[146]],[face_landmarks.landmark[91]], [face_landmarks.landmark[181]], [face_landmarks.landmark[84]],[face_landmarks.landmark[17]],[face_landmarks.landmark[314]], [face_landmarks.landmark[405]], [face_landmarks.landmark[321]], [face_landmarks.landmark[375]], [face_landmarks.landmark[291]])),
                "facepos": liste,
            }

        self._ready = True  # boolean si visage existe
        self._faces = rects  # Tous les visages détectés
        self._face = point_dict(large,haut)  # dictionnaire des face_landmarks.landmark du visage le plus grand
        # print("face : ", self._face)
        #self._rect = rects  # Contour du visage le plus grand
        self.extract_vector(large,haut)

    def define_vector(self, salient):
        interest = []
        coefs = []
        nbFeatures = 0
        if salient[1] == '1':
            add = ["left_eyebrow", "right_eyebrow"]
            interest.extend(add)
            nbFeatures += 24
            coef = 24 * [24]
            coefs.extend(coef)
        if salient[3] == '1':
            add = ["nose_tip", "nose_bridge"]
            interest.extend(add)
            nbFeatures += 24
            coef = 24 * [24]
            coefs.extend(coef)
        if salient[0] == '1':
            add = ["left_eye", "right_eye"]
            interest.extend(add)
            nbFeatures += 56
            coef = 56 * [56]
            coefs.extend(coef)
        if salient[2] == '1':
            add = ["bottom_lip", "top_lip"]
            interest.extend(add)
            nbFeatures += 44  #
            coef = 44 * [44]
            coefs.extend(coef)


        self._interest = interest
        self._sizeData = nbFeatures
        self._coefs = np.array(coefs)
        print("\t profils : ", self._interest)
        print("\t nombre de face_landmarks.landmark : ", self.sizeData)

    def extract_vector(self,large,haut):              #Pour demain
        ## Normaliser les face_landmarks.landmark...
        # Point de référence du visage

        refx = self._face["facepos"][0][0]
        refy = self._face["facepos"][0][1]
        largx = self._face["facepos"][0][2] - refx
        largy = self._face["facepos"][0][3] - refy
        # print(face)
        result = []

        for k, pt in self._face.items():
            if k == "facepos":
                continue
                # if "lip" in k: 4 juin 2021 => reduct
            if "lip" in k:

                # lire tableau et le reverser dans result
                for coord in pt:
                    if k in self._interest:

                        px = (coord.x * large - refx) / largx
                        py = (coord.y * haut - refy) / largy
                        result = np.append(result, px)
                        result = np.append(result, py)
                continue

            if k in self._interest:
                for coord in pt:
                    # print("k :",k)
                    px = (coord.x * large - refx) / largx
                    py = (coord.y * haut - refy) / largy
                    result.append(px)
                    result.append(py)
        # print("len(result) : ", len(result))
        # print()
        # print()
        self._current = np.array(result)  # face_landmarks.landmark d'intérêts


    def insert_capture(self, frame):
        """
        Intégrer l'image des face_landmarks.landmark pris en compte dans l'image
        :param frame: l'image
        :param vector : les saillances apprises
        :return: l'image modifie
        """
        vector = self._current
        coin_gauche = (20, 0)
        largeur = 100
        neurframe = np.full((largeur, largeur, 3), 200, np.uint8)

        cv2.rectangle(neurframe, (0, 0), (largeur - 1, largeur - 1), (0, 0, 0))
        i = 0
        while i < len(vector):
            px = int(round(largeur * vector[i]))
            py = int(round(largeur * vector[i + 1]))
            cv2.circle(neurframe, (px, py), 1, 100, -1)
            i = i + 2
        frame[coin_gauche[0]:coin_gauche[0] + largeur, coin_gauche[1]:coin_gauche[1] + largeur] = neurframe

        return frame

    def insert_salient(self, frame, winner):
        """
        dessiner les face_landmarks.landmark de saillances, le cadre du visage et le cluster
        :param frame: l'image
        :param face: les face_landmarks.landmark de saillances
        :param cluster: le resultat de clustering
        :return: l'image modifie
      """
        face = self._face

        # refx = face["facepos"][0][0]
        # refy = face["facepos"][0][1]
        # largx = face["facepos"][0][2] - face["facepos"][0][0]
        # largy = face["facepos"][0][3] - face["facepos"][0][1]
        largeur_cv = frame.shape[1] # 400 pixels
        longueur_cv = frame.shape[0]

        for k, pt in face.items():
            if k == "facepos":
                # rectangle pour entourer le visage
                [(x1, y1, x2, y2)] = pt
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))
                cv2.rectangle(frame, (x1, y1), (x1 + 100, y1 - 18), (0, 0, 255), -1)
                cv2.putText(frame, "Neuron({0},{1})".format(winner[0], winner[1]),
                            (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                continue
            for point in pt:

                x = int(point.x * largeur_cv)
                y = int(point.y * longueur_cv)
                cv2.circle(frame, (x,y), 1, landmarks_mp.COLORS[k], -1)
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

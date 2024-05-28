import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from imutils import face_utils
from numpy import concatenate as cat

class landmarks_corps:
    COLORS = {'body': [0,0,0]}


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
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
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
        rects = self.pose.process(img)

        haut, large = img.shape[0], img.shape[1]
        results = self.detector.process(img)

        # on sort si aucun visage DETECTE
        if not rects.pose_landmarks:
            self._ready = False
            self._faces = None
            self._face = None
            self._rect = None
            return
        else:
            # extraction des face_landmarks.landmark de saillances
            if(rects.pose_landmarks != None):
                list_body = rects.pose_landmarks.landmark
            else: return

        def point_dict(large,haut,bool=True):
            return {
                "facepos": [(int(results.detections[0].location_data.relative_bounding_box.xmin * img.shape[1]),
                             int(results.detections[0].location_data.relative_bounding_box.ymin * img.shape[0]),
                             int(results.detections[0].location_data.relative_bounding_box.width * img.shape[1]) + int(
                                 results.detections[0].location_data.relative_bounding_box.xmin * img.shape[1]),
                             int(results.detections[0].location_data.relative_bounding_box.height * img.shape[0]) + int(
                                 results.detections[0].location_data.relative_bounding_box.ymin * img.shape[0]))],
                "body": list_body[11:17],
            }

        self._ready = True  # boolean si visage existe
        self._faces = rects  # Tous les visages détectés
        self._face = point_dict(large,haut)  # dictionnaire des face_landmarks.landmark du visage le plus grand
        self._body = point_dict(large,haut)

        self.extract_vector(large,haut)

    def define_vector(self, salient):
        interest = []
        coefs = []
        nbFeatures = 0
        if salient[1] == '1':
            add = ["body"]
            interest.extend(add)
            nbFeatures += 12
            coef = 12 * [12]
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

        for k, pt in self._body.items():
            if k == "facepos":
                continue
            for coord in pt:
                px = (coord.x * large - refx) / largx
                py = (coord.y * haut - refy) / largy
                result.append(px)
                result.append(py)
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
        body = self._body

        largeur_cv = frame.shape[1] # 400 pixels
        longueur_cv = frame.shape[0]

        for k, pt in body.items():
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
                cv2.circle(frame, (x,y), 1, landmarks_corps.COLORS[k], -1)
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

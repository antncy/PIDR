# -*- coding: utf-8 -*-

"""
CameraFrame est le repère dans lequel on calcule les images de la caméro pour la
lampe simulée.

Cela permet de:
- calculer la position, dans l'image, d'un pseudo-visage
- d'afficher la CameraFrame dans un plot
- d'afficher une sorte de champ de vue de la caméra
- d'afficher comment un pseudo-visage est perçu
"""

import ikpy.utils.geometry as ikg
import ikpy.utils.plot as ikp
import numpy as np

# ******************************************************************************
# ****************************************************************** CameraFrame
# ******************************************************************************
class CameraFrame(object):
    """
    - Tout ce qui se termine en _h est en coordonnées homogènes
    - Tout ce qui comment par t est tranformé par la tranform
    """

    # ****************************************************************** __init__
    def __init__(self):
        """Définit origine 'o' et vecteurs unitaires 'u,v,w' du repère"""
        self.o_h = ikg.cartesian_to_homogeneous_vectors( np.array([0, 0, 0]) )
        self.u_h = ikg.cartesian_to_homogeneous_vectors( np.array([1, 0, 0]) )
        self.v_h = ikg.cartesian_to_homogeneous_vectors( np.array([0, 1, 0]) )
        self.w_h = ikg.cartesian_to_homogeneous_vectors( np.array([0, 0, 1]) )

        # Par défaut, le repère est transformé avec l'identité
        self.to_h = self.o_h
        self.tu_h = self.u_h
        self.tv_h = self.v_h
        self.tw_h = self.w_h

        # Des points pour dessiner le 'field' de la CameraFrame
        # Un rectangle parallèle au plan Oxy (de l'image camera),
        # de taille (xsize,ysize), situé à une distance zdist vers "l'avant"
        # et un point au centre de l'image
        xsize = 0.4
        ysize = 0.3
        zdist = 0.1
        pts = [[xsize/2, ysize/2, -zdist],
               [-xsize/2, ysize/2, -zdist],
               [-xsize/2, -ysize/2, -zdist],
               [xsize/2, -ysize/2, -zdist],
               [0, 0, 0],                 # origin point
               [0, 0, -zdist]]  # up point
        self.pts_field_h = [ikg.cartesian_to_homogeneous_vectors(np.array(p)) for p in pts]
        self.tpts_field_h = [p for p in self.pts_field_h]
        # Besoin d'une transformation pour passer du repère CameraFrame à
        # un repère pour plotter le Field
        # rotation de -PI/2 selon Oz local
        zt = ikg.rz_matrix( - np.pi/2 )
        hzt = ikg.to_transformation_matrix( translation=np.array( [0, 0, 0] ),
                                            orientation_matrix=zt)
        # puis rotation de PI/2 selont Ox local
        xt = ikg.rx_matrix( np.pi/2 )
        hxt = ikg.to_transformation_matrix( translation=np.array( [0, 0, 0] ),
                                            orientation_matrix=xt)
        self.trans_camera_to_field = np.dot(hzt, hxt)

    # ******************************************************************** update
    def update(self, transform):
        """Calcule la CameraFrame transformée, et les points du Field qui doivent
        être de plus transformé par camera_to_field"""
        self.to_h = np.dot( transform, self.o_h )
        self.tu_h = np.dot( transform, self.u_h )
        self.tv_h = np.dot( transform, self.v_h )
        self.tw_h = np.dot( transform, self.w_h )

        field_trans = np.dot(transform, self.trans_camera_to_field)
        self.tpts_field_h = [np.dot(field_trans, p) for p in self.pts_field_h]
        # self.tpts_field_h = [np.dot(np.dot(transform, p), self.trans_camera_to_field)
        #                      for p in self.pts_field_h]

    # ************************************************************ coord_imgframe
    def coord_imgframe(self, pos):
        """Calcule les coordonnées dans le repère image, sachant que dans ce repère,
        l'image se calcule dans les axes -Ov_h, Ow_h.

        Return: np.array( [coord en X image, coord en Y image] )
        """
        vx_img = -ikg.homogeneous_to_cartesian_vectors( self.tv_h - self.to_h )
        vy_img = ikg.homogeneous_to_cartesian_vectors( self.tw_h - self.to_h )
        # print(f"vx_img={vx_img}, lengh={np.linalg.norm(vx_img)}")

        pos_o = pos - ikg.homogeneous_to_cartesian_vectors( self.to_h  )
        # projette sur les vecteurs unitaires de l'image
        fx_img = np.dot(vx_img, pos_o)
        fy_img = np.dot(vy_img, pos_o)
        # print(f"In the image, pos at {fx_img} x {fy_img}")

        return np.array([fx_img, fy_img])

    # ***************************************************************************
    # ****************************************************************** plotting
    # ***************************************************************************
    def plot(self, axes, scale=0.25):
        """Plot le repère transformé, chaque axe est un segment de longueur `scale`
        rouge pour Ou, vert pour Ov, bleu pour Ow."""

        for pt,col in [(self.tu_h, "r-"), (self.tv_h, "g-"), (self.tw_h, "b-")]:
            axes.plot( [self.to_h[0], pt[0]],
                       [self.to_h[1], pt[1]],
                       [self.to_h[2], pt[2]],
                       col )

    def plot_projected_point(self, axes, pos):
        """Plot le point (petite boule cyan) et la projection (ligne pointillée cyan)
        jusqu'à l'origine de la CameraFrame transformée"""

        axes.scatter( [pos[0]], [pos[1]], [pos[2]], s=55, c="c" )
        axes.plot( [self.to_h[0], pos[0]], [self.to_h[1], pos[1]], [self.to_h[2], pos[2]], "c--" )

    def plot_field(self, axes):
        # first the "cadre" and "central" part
        for pt1,pt2 in [(0,1), (1,2), (2,3), (3,0), (0,5), (5,1)]:
            axes.plot( [self.tpts_field_h[pt1][0], self.tpts_field_h[pt2][0]],
                       [self.tpts_field_h[pt1][1], self.tpts_field_h[pt2][1]],
                       [self.tpts_field_h[pt1][2], self.tpts_field_h[pt2][2]],
                       "b-")
            # then linked to origin of camera
            for pt in self.tpts_field_h[0:4]:
                axes.plot( [self.tpts_field_h[4][0], pt[0]], [self.tpts_field_h[4][1], pt[1]],
                           [self.tpts_field_h[4][2], pt[2]], "r-")

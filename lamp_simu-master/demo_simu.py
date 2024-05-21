#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Démo de modèle directe et inverse de la lampe en utilisant ikpy
## avec python3
## Il faut avoir installé ikpy (pip install ikpy)

from ikpy.chain import Chain
from ikpy.link import OriginLink
import ikpy.utils.geometry as ikg
import ikpy.utils.plot as ikp
from ikpy.utils.plot import init_3d_figure, show_figure

from parallel_link import ParallelLink
from camera_frame import CameraFrame

import numpy as np

# ******************************************************************** Géométrie
## Géométrie de la lampe
## Ox vers l'avant, Oy vers la gauche, Oz vers le haut
base_rot_pos   = np.array( [0,0,0.0715] )
base_rot_orient= np.array( [0,0,0] )
base_rot_axe   = np.array( [0,0,1] )
base_arm_pos   = np.array( [0.0223,-0.0185,0.1025] )
base_arm_orient= np.array( [0,0,0] )
base_arm_axe   = np.array( [0,1,0] )   
arm_pos        = np.array( [0.0224,-0.0185,0.3276] )
arm_orient     = np.array( [0,0,0] )
arm_axe        = np.array( [0,1,0] )
head_arm_pos   = np.array( [0.1933,-0.0185,0.3051] )
head_arm_orient= np.array( [0,0,0] )
head_arm_axe   = np.array( [0,1,0] )
head_pos       = np.array( [0.1933,-0.0085,0.3871] )
head_orient    = np.array( [0,0,0] )
head_axe       = np.array( [0,0,1] )
box_pos        = np.array( [0.2083,-0.0085, 0.3806])
bulb_pos_rel   = np.array( [0.05,0,0] )

# ****************************************************************** create_lamp
def create_lamp():
    """
    Cree la lampe comme une chaîne cinématique de ParallelLink. (ik.Chain)
    Les Link où je ne préciser pas de `prev_ink` sont en fait des Link classiques.
    """
    ## Création d'un Link Origin : base du robot
    link_0 = OriginLink()

    ## Link_1 : mot_base
    link_1 = ParallelLink( name="l_base_rot",
                           translation_vector = base_rot_pos,
                           orientation = base_rot_orient,
                           rotation = base_rot_axe )

    ## Link_2 : base_arm
    link_2 = ParallelLink( name="l_base_arm",
                           translation_vector = base_arm_pos - base_rot_pos,
                           orientation = base_arm_orient,
                           rotation = base_arm_axe )

    ## Link_3 : arm
    link_3 = ParallelLink( name="l_arm",
                           translation_vector = arm_pos - base_arm_pos,
                           orientation = arm_orient,
                           rotation = arm_axe,
                           prev_link = link_2 )

    ## Link_4 : head_arm
    link_4 = ParallelLink( name="l_head_arm",
                           translation_vector = head_arm_pos - arm_pos,
                           orientation = head_arm_orient,
                           rotation = head_arm_axe,
                           prev_link = link_3)
    ## Link 5 : head
    link_5 = ParallelLink( name="l_head",
                           translation_vector = head_pos - head_arm_pos,
                           orientation = head_orient,
                           rotation = head_axe )
    ## Link 6 : bulb
    link_6 = ParallelLink( name="l_bulb",
                           translation_vector = box_pos - head_pos + bulb_pos_rel,
                           orientation = head_orient,
                           rotation = head_axe )
    ## Une chaine avec ces link
    # première liste : les Link
    # deuxième liste : si ces Link sont actifs
    lamp_chain = Chain( [link_0, link_1, link_2, link_3, link_4, link_5, link_6],
                        [False, True, True, True, True, True, False],
                        name = "lamp" )
    return lamp_chain


def plot_camera_field(transform, xsize=0.4, ysize=0.3, zdist=0.2):
    """Draw a rectable XxY at distance zdist, linked to origin point
    "Up" is indicated with a small triangle"""

    pts = [[xsize/2, ysize/2, -zdist],
           [-xsize/2, ysize/2, -zdist],
           [-xsize/2, -ysize/2, -zdist],
           [xsize/2, -ysize/2, -zdist],
           [0, 0, 0],                 # origin point
           [0, 0, -zdist]]  # up point

    # transformed pts
    thpts = [np.dot(transform, ikg.cartesian_to_homogeneous_vectors(np.array(vec)))
                    for vec in pts]

    # first the "cadre" and "upper" part
    for pt1,pt2 in [(0,1), (1,2), (2,3), (3,0), (0,5), (5,1)]:
        ax.plot( [thpts[pt1][0], thpts[pt2][0]], [thpts[pt1][1], thpts[pt2][1]],
                 [thpts[pt1][2], thpts[pt2][2]], "b-")
    # then linked to origin of camera
    for pt in thpts[0:4]:
        ax.plot( [thpts[4][0], pt[0]], [thpts[4][1], pt[1]],
                 [thpts[4][2], pt[2]], "r-")

def camera_transform():
    """Return a camera transform so that image is XxY and -Z is depth"""
    # Rotation around -pi/2 around Oz
    zt = ikg.rz_matrix( - np.pi/2 )
    hzt = ikg.to_transformation_matrix( translation=np.array( [0, 0, 0] ),
                                        orientation_matrix=zt)
    # then around Ox
    xt = ikg.rx_matrix( np.pi/2 )
    hxt = ikg.to_transformation_matrix( translation=np.array( [0, 0, 0] ),
                                        orientation_matrix=xt)
    hbt = np.dot(hzt, hxt)
    return hbt

def plot_camera_frame( ax, transform, scale=0.25, face_pos=None ):
    # get the transform of the origin_pt and of the unitary vectors
    oh = ikg.cartesian_to_homogeneous_vectors( np.array([0, 0, 0]) )
    uh = ikg.cartesian_to_homogeneous_vectors( np.array([1, 0, 0]) )
    print(f"uh={uh}")
    vh = ikg.cartesian_to_homogeneous_vectors( np.array([0, 1, 0]) )
    wh = ikg.cartesian_to_homogeneous_vectors( np.array([0, 0, 1]) )

    toh = np.dot(transform, oh)
    tuh = np.dot(transform, uh)
    tvh = np.dot(transform, vh)
    twh = np.dot(transform, wh)

    print(f"toh={toh}")
    print(f"tuh={tuh}")
    ax.plot( [toh[0], tuh[0]], [toh[1], tuh[1]], [toh[2], tuh[2]], "r-" )
    ax.plot( [toh[0], tvh[0]], [toh[1], tvh[1]], [toh[2], tvh[2]], "g-" )
    ax.plot( [toh[0], twh[0]], [toh[1], twh[1]], [toh[2], twh[2]], "b-" )

    if face_pos is not None:
        # dans ce repère, l'image caméra se trouve dans les axes -Oy, Oz
        vx_img = -ikg.homogeneous_to_cartesian_vectors( tvh - toh )
        vy_img = ikg.homogeneous_to_cartesian_vectors( twh - toh )
        print(f"vx_img={vx_img}, lengh={np.linalg.norm(vx_img)}")

        face_oh = face_pos - ikg.homogeneous_to_cartesian_vectors( toh  )
        fx_img = np.dot(vx_img, face_oh)
        fy_img = np.dot(vy_img, face_oh)
        print(f"In the image, face at {fx_img} x {fy_img}")

        ax.scatter( [face_pos[0]], [face_pos[1]], [face_pos[2]], s=55, c="c" )
        ax.plot( [toh[0], face_pos[0]], [toh[1], face_pos[1]], [toh[2], face_pos[2]], "c--" )
# ************************************************************************* Démo
if __name__ == '__main__':
    ## Crée une chaine cinémtique
    ## Attention, le moteur '0' n'est pas controllable, c'est le point fixe
    ## de la lampe. De même, le moteur 6 est l'extrémité, pas de vrai moteur.
    lamp_chain = create_lamp()

    ## Applique des commandes (position en angle) sur chaque moteur
    print( "**** Mode FORWARD **** : commande directe" )
    # angle_cmd = np.radians( [0, 20, 0, 0, -20, 30, 0] )
    angle_cmd = np.radians( [0, 0, 0, 0, 0, 0, 0] )
    transform = lamp_chain.forward_kinematics( angle_cmd,
                                               full_kinematics=False)
    ## Le résultat est une matrice de Transformation
    print( "TF",transform)
    print( "Extrémité (x,y,z) en mètre = {}".format( transform[0:3,3] ))

    # un pseudo visage
    face_pt = np.array( [1.0, 0.5, 0.5] )

    ## Une CameraFrame pour calculée des coordonnées dans le référenciel caméra
    ## et afficher ce dernier
    camera_ref = CameraFrame()
    # on position ce référenciel un utilisant le transform de l'extrémité de la lampe
    camera_ref.update( transform )

    ## On peut afficher la configuration de la lampe
    # Crée un plot, affiche le plot
    fig, ax = init_3d_figure()
    fig.tight_layout()

    # la commande qui suit ne permet que d'afficher la lampe, et rien d'autre
    # lamp_chain.plot( angle_cmd, ax, target=None, show=True )

    # Ici, on affiche d'abord la Lampe
    ikp.plot_chain(lamp_chain, angle_cmd, ax, name="Lamp")

    # afficher le référenciel caméra (3 segment R, G, B)
    camera_ref.plot(ax)
    camera_ref.plot_field(ax)
    camera_ref.plot_projected_point(ax, face_pt)
    # les coordonnées de la pseudo-face dans le repère image
    face_img = camera_ref.coord_imgframe(face_pt)
    print(f"La pseudo-face est VUE en {face_img}")

    # et il faut penser à montrer tout ça
    ikp.show_figure()

    print("Done")

    # FIXME TODO voir si/comment inverse_kinematics à changé entre les versions
    # de ikpy !!!!

    # ## Trouve des commandes permettant d'accéder à une position donnée
    # ## ATTENTION : par défaut, quand ikpy s'intéresse aussi à l'orientation,
    # # il fait uniquement des différences de matrice de transformation
    # # et ne fait pas de différence entre des distances (mètre) ou
    # # des angles (radians)
    # ## PLUS d'INFO : voir ikpy.chain.py et ikpy.inverse_kinematics.py
    # print( "\n**** MODE INVERSE **** : commance inverse" )
    # print( "  Je mets la target un peu loin, exprès " )
    # target_pos = np.array( [0.4, -0.3, 0.45] )
    # target_cmd = lamp_chain.inverse_kinematics( target_position=target_pos,
    #                                             target_orientation=None,
    #                                             orientation_mode=None )
    # print( "target_cmd=",np.degrees(target_cmd) )
    # # Crée un plot, affiche le plot
    # fig, ax = init_3d_figure()
    # lamp_chain.plot( target_cmd, ax, target=target_pos, show=True )

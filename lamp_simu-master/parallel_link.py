# -*- coding: utf-8 -*-

# Definition of ParallelLink

from ikpy.link import Link
import ikpy.utils.geometry as ikgeom
import numpy as np

# ******************************************************************************
# ***************************************************************** ParallelLink
# ******************************************************************************
class ParallelLink(Link):
    """
    Parallel Link heavily based on URDFLink
    If a 'prev_link' is given, then behave as a true Parallel Link
    otherwise, it is a "classic" link (not parallel).
    """

    # ------------------------------------------------------- ParallelLink::init
    def __init__(self, name, translation_vector, orientation, rotation, bounds=(None, None), prev_link=None):
        """
        
        Params:
        - `self`:
        - `name`:
        - `translation_vector`:
        - `orientation`:
        - `rotation`:
        - `bounds`:
        - `prev_link`:
        """
        Link.__init__(self, name=name, length=0.0, bounds=bounds)
        self.translation_vector = np.array(translation_vector)
        self.orientation = np.array(orientation)
        self.rotation = np.array(rotation)
        self.prev_link = prev_link
        self.theta = 0

        # TODO attention, "hack"
        self.has_rotation = True
        self.has_translation = False

        self._length = np.linalg.norm(translation_vector)
        self._axis_length = self._length
    # -------------------------------------------------------- ParallelLink::str
    def __str__(self):
        return("""Parallel Link {} :
        Constrained by : {}
        Translation : {}
        Orientation : {}
        Rotation : {}""".format(self.name, self.prev_link.name, self.translation_vector, self.orientation, self.rotation))
    # ----------------------------------------- ParallelLink::_get_rotation_axis
    def _get_rotation_axis(self):
        return (np.dot(ikgeom.homogeneous_translation_matrix(*self.translation_vector), np.dot(ikgeom.cartesian_to_homogeneous(ikgeom.rpy_matrix(*self.orientation)), ikgeom.cartesian_to_homogeneous_vectors(self.rotation * self._axis_length))))
    def get_rotation_axis(self):
        return self._get_rotation_axis()
    # ---------------------------------------- ParallelLink::get_rotation_matrix
    def get_rotation_matrix(self, theta):
         # Init the rotation matrix
        frame_matrix = np.eye(4)
        # Apply orientation
        frame_matrix = np.dot(frame_matrix, ikgeom.cartesian_to_homogeneous(ikgeom.rpy_matrix(*self.orientation)))
        # Apply rotation matrix
        frame_matrix = np.dot(frame_matrix, ikgeom.cartesian_to_homogeneous(ikgeom.axis_rotation_matrix(self.rotation, theta)))

        return frame_matrix
    # ----------------------------------- ParallelLin::get_transformation_matrix
    def get_transformation_matrix(self, theta):
        # memorise theta
        self.theta = theta
        # Constraint
        if self.prev_link:
            # get previous rotation matrix
            prev_rot = self.prev_link.get_rotation_matrix(self.prev_link.theta)
            # inverse rotation
            prev_rot = np.transpose( prev_rot )
            
        # Init the transformation matrix
        frame_matrix = np.eye(4)

        # First, apply translation matrix
        frame_matrix = np.dot(frame_matrix, ikgeom.homogeneous_translation_matrix(*self.translation_vector))

        # Then, if has previous, unapply rotation
        if self.prev_link:
            frame_matrix = np.dot(frame_matrix, prev_rot )

        # Apply orientation
        frame_matrix = np.dot(frame_matrix, ikgeom.cartesian_to_homogeneous(ikgeom.rpy_matrix(*self.orientation)))

        # Apply rotation matrix
        frame_matrix = np.dot(frame_matrix, ikgeom.cartesian_to_homogeneous(ikgeom.axis_rotation_matrix(self.rotation, theta)))

        return frame_matrix
    # ---------------------------------------------------- get_link_frame_matrix
    def get_link_frame_matrix(self, parameters):

        # DEBUG
        # print( f"get_link_frame_matrix({parameters})" )

        # DEL theta = parameters["theta"]
        theta = parameters
        return self.get_transformation_matrix(theta)
# ******************************************************************************

from math import sqrt
class distance:

    def cartesienne(self,vect1,vect2):
        dist=0
        for i in range(1,len(vect1)):
            dist=dist+sqrt( (vect1[i]-vect2[i])*(vect1[i]-vect2[i]) )
        return dist

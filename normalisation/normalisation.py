class normalisation:
    def __init__(self):
        return None

    def moynormalise(self,vecteur):
        moy=0.0
        for elem in vecteur:
            moy=moy+elem

        res=[]
        for elem in vecteur:
            res.append(elem/moy)

        return res

    def moynormalise2D(self, vecteur):
        moy =(0.0,0.0)
        for elem in vecteur:
            moy=(moy[0]+elem[0] ,moy[1]+elem[1])
        res = []
        for elem in vecteur:
            res.append(( elem[0] / moy[0] , elem[1] / moy[1] ))
        return res

    def maxnormalise(self, vecteur):
        maxi=max(vecteur)
        res=[]
        for elem in vecteur:
            res.append(elem/maxi)

        return res

    def val01(self,x):
        v= 1-(1/float(x+0.00000000000000001))
        if(v<0):
            return 0
        if(v>1):
            return 1
        return v
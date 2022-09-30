class vehicule:

    def __init__(self, couleur='noire', vitesse=0, roues=4):
        self.couleur = couleur
        self.vitesse = vitesse
        self.roues = roues

    def accelerer(self, vitesse):
        self.vitesse += vitesse

    def stop(self):
        self.vitesse = 0

    def afficher(self):
        print(f'couleur: {self.couleur}\nroues: {self.roues}\nvitesse: {self.vitesse}')

voiture_1 = vehicule(couleur='rouge')
voiture_1.accelerer(100)
voiture_1.afficher()
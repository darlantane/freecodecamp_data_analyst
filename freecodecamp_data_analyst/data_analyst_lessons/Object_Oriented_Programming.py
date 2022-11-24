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

class voiture_electrique(vehicule):
    """
    La classe moto hérite des méthodes et des attributs de la classe véhicule
    """

    def __init__(self, couleur='black', vitesse=0, roues=4, autonomie=100):
        super().__init__(couleur, vitesse, roues) # super() permet d'utiliser la fonction de la classe "parent"
        self.autonomie = autonomie

    # Ré-écriture de certaines méthodes
    def accelerer(self, vitesse):
        super().accelerer(vitesse)
        self.autonomie -= 0.1 *self.vitesse

    def afficher(self):
        super().afficher()
        print(f'autonomie: {self.autonomie}')
voiture_2 = voiture_electrique()
voiture_2.afficher()

voiture_2.accelerer(10)
voiture_2.afficher()
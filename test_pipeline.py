import unittest
import pandas as pd
from pipline import split_scale  
class TestPipeline(unittest.TestCase):

    def setUp(self):
        """Préparer les données et le pipeline"""
        self.X_train, self.X_test, self.y_train, self.y_test = split_scale()

    def test_no_missing_values(self):
        """Vérifier qu'il n'y a pas de valeurs manquantes dans X et y"""
        self.assertFalse(pd.isnull(self.X_train).any(), "X_train contient des valeurs manquantes")
        self.assertFalse(pd.isnull(self.X_test).any(), "X_test contient des valeurs manquantes")
        self.assertFalse(pd.isnull(self.y_train).any(), "y_train contient des valeurs manquantes")
        self.assertFalse(pd.isnull(self.y_test).any(), "y_test contient des valeurs manquantes")

    def test_dimensions(self):
        """Vérifier que les dimensions correspondent"""
        # X doit avoir autant de lignes que y
        self.assertEqual(self.X_train.shape[0], self.y_train.shape[0], "X_train et y_train n'ont pas le même nombre de lignes")
        self.assertEqual(self.X_test.shape[0], self.y_test.shape[0], "X_test et y_test n'ont pas le même nombre de lignes")

        # X_train et X_test doivent avoir le même nombre de colonnes après VarianceThreshold
        self.assertEqual(self.X_train.shape[1], self.X_test.shape[1], "X_train et X_test n'ont pas le même nombre de colonnes")

if __name__ == "__main__":
    unittest.main()

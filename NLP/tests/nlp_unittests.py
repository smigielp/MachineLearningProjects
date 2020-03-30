import unittest
import nlp_models.preprocessing as prep


class TestNLP(unittest.TestCase):

    def test_review_cleanup(self):
        review = 'The dishes were very tasty'
        clean_review = prep.review_cleanup(review)
        self.assertEqual(clean_review, 'dish tasti')


if __name__ == '__main__':
    unittest.main()

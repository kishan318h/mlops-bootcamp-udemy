import unittest
import app

class BasicTestCase(unittest.TestCase):
    def test_hello_world(self):
        tester = app.app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Hello', response.data)

if __name__ == '__main__':
    unittest.main()
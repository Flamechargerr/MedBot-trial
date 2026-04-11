import importlib
import sys
import types
import unittest


def _install_flask_stub(path='/api/v1/chat', method='POST', auth_header=''):
    flask_module = types.ModuleType("flask")

    flask_module.request = types.SimpleNamespace(
        path=path,
        method=method,
        headers={'Authorization': auth_header},
        content_length=0,
        is_json=True,
        remote_addr='127.0.0.1',
    )
    flask_module.jsonify = lambda payload: payload
    sys.modules['flask'] = flask_module


class SecurityGuardTests(unittest.TestCase):
    def setUp(self):
        _install_flask_stub()
        self.security_module = importlib.reload(importlib.import_module('src.security'))

        class Config:
            app_auth_token = 'token-123'
            max_request_bytes = 10
            rate_limit_per_minute = 2
            cors_origins = ['http://localhost:5000']

        self.guard = self.security_module.RequestGuard(Config())

    def test_auth_rejected_when_missing_token(self):
        result = self.guard.enforce_auth()
        self.assertIsNotNone(result)
        payload, status = result
        self.assertEqual(status, 401)
        self.assertEqual(payload['status'], 'error')

    def test_request_size_rejected_when_too_large(self):
        self.security_module.request.content_length = 100
        result = self.guard.enforce_request_size()
        self.assertIsNotNone(result)
        payload, status = result
        self.assertEqual(status, 413)
        self.assertEqual(payload['status'], 'error')

    def test_rate_limit_exceeded(self):
        self.security_module.request.headers['Authorization'] = 'Bearer token-123'
        self.assertIsNone(self.guard.enforce_rate_limit())
        self.assertIsNone(self.guard.enforce_rate_limit())
        result = self.guard.enforce_rate_limit()
        self.assertIsNotNone(result)
        payload, status = result
        self.assertEqual(status, 429)
        self.assertEqual(payload['status'], 'error')


if __name__ == '__main__':
    unittest.main()

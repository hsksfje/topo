"""Frozen transport/retry method with only explicit effort and receipt-mapping knobs.
The original immutable API package and all numerical code remain unchanged.
"""
import json,time,shutil
from urllib.parse import urlparse
from initskill.io import dump,digest
import portable_lock as fcntl
from api_client import Client as BaseClient,read_credentials

class EffortClient(BaseClient):

    def call(self, job_id, messages, response_format, max_tokens=8192):
        payload = dict(model=self.config['model'], messages=messages, temperature=0.2, max_tokens=self.config['reasoning_max_tokens'], response_format=response_format, reasoning_effort=self.config.get('api_reasoning_effort', 'medium'), extra_body={'thinking': {'type': 'enabled'}})
        identity = digest(payload)
        folder = self.output / 'requests' / job_id
        if (folder / 'attempt.json').exists() and (folder / 'receipt.json').exists():
            if json.loads((folder / 'attempt.json').read_text(encoding='utf-8'))['identity'] != identity:
                raise ValueError('Request changed')
            if json.loads((folder / 'receipt.json').read_text(encoding='utf-8'))['status'] in ('received', 'invalid_response'):
                return json.loads((folder / 'parsed.json').read_text(encoding='utf-8'))
        retries = len(list((folder / 'retry_history').glob('attempt_*')))
        while True:
            self._wait()
            with self._ledger_lock, (self.output / '.api.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                attempt = folder / 'attempt.json'
                if attempt.exists():
                    if json.loads(attempt.read_text(encoding='utf-8'))['identity'] != identity:
                        raise ValueError('Request changed')
                    receipt_path = folder / 'receipt.json'
                    if not receipt_path.exists():
                        raise RuntimeError('Unsettled request; inspect outcome before resend')
                    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                    status = receipt['status']
                    if status in ('received', 'invalid_response'):
                        return json.loads((folder / 'parsed.json').read_text(encoding='utf-8'))
                    if not self._retryable(receipt, folder):
                        raise RuntimeError('Permanent or unknown API failure; inspect receipt')
                    if retries >= self.MAX_RETRIES:
                        raise RuntimeError('API retry limit reached; inspect retry_history')
                    due = receipt.get('retry_not_before', 0)
                    if due > self._clock():
                        self._schedule(job_id, due - self._clock(), receipt.get('error_type'))
                        continue
                    self._check_pause()
                    if self.budget()['attempts'] >= self.config['max_api_calls']:
                        raise RuntimeError('Request cap including retries')
                    archive = folder / 'retry_history' / ('attempt_%04d' % (retries + 1))
                    archive.mkdir(parents=True, exist_ok=False)
                    for name in ('request.json', 'attempt.json', 'receipt.json', 'parsed.json', 'http_error.json', 'response.json'):
                        if (folder / name).exists():
                            shutil.copy2(folder / name, archive / name)
                    for name in ('attempt.json', 'receipt.json', 'parsed.json', 'http_error.json', 'response.json'):
                        (folder / name).unlink(missing_ok=True)
                    retries += 1
                self._check_pause()
                size = len(json.dumps(payload, ensure_ascii=False).encode())
                budget = self.budget()
                if size > self.config['max_input_bytes'] or budget['attempts'] >= self.config['max_api_calls']:
                    raise RuntimeError('Request cap')
                dump(folder / 'request.json', payload)
                dump(folder / 'attempt.json', dict(identity=identity, started_unix=self._clock(), reserved_completion_tokens=payload['max_tokens'], retry_number=retries))
            started = time.monotonic()
            raw = {}
            parsed = None
            error = None
            status = 'transport_error'
            retryable = False
            delay = 0
            code = None
            try:
                with self._backend_lock:
                    if self.backend is None:
                        from openai import OpenAI, DefaultHttpxClient
                        c = read_credentials(self.env_file)
                        if urlparse(c['MODEL_API_BASE_URL']).hostname != self.config['provider_host']:
                            raise ValueError('Provider changed')
                        self.backend = OpenAI(api_key=c['MODEL_API_KEY'], base_url=c['MODEL_API_BASE_URL'], max_retries=0, timeout=600, http_client=DefaultHttpxClient(follow_redirects=False))
                response = self.backend.chat.completions.create(**payload)
                raw = response.model_dump(mode='json')
                dump(folder / 'response.json', raw)
                if raw.get('model') != self.config['model'] or response.choices[0].finish_reason != 'stop':
                    raise ValueError('Model mismatch or truncated output')

                def nonfinite(value):
                    raise ValueError('Nonfinite JSON constant')
                parsed = json.loads(response.choices[0].message.content or '', parse_constant=nonfinite)
                status = 'received'
            except Exception as exc:
                code = getattr(exc, 'status_code', None)
                error = type(exc).__name__
                status = 'invalid_response' if raw else 'rejected' if code is not None else 'transport_error'
                retryable = not raw and (code in (408, 409, 429) or (isinstance(code, int) and code >= 500) or error in ('APITimeoutError', 'APIConnectionError', 'TimeoutError', 'ConnectionError'))
                if code:
                    dump(folder / 'http_error.json', dict(status_code=code, error_type=error, message=str(getattr(exc, 'message', ''))[:1500]))
                if retryable:
                    delay = self._retry_delay(exc, retries)
            dump(folder / 'parsed.json', parsed)
            usage = raw.get('usage') or {}
            details = usage.get('completion_tokens_details') or {}
            dump(folder / 'receipt.json', dict(status=status, error_type=error, usage=usage, reasoning_tokens=details.get('reasoning_tokens'), requested_effort=self.config.get('api_reasoning_effort', 'medium'), documented_provider_mapping=self.config.get('provider_reasoning_mapping', 'low'), thinking='enabled', seconds=time.monotonic() - started, response_id=raw.get('id'), model=raw.get('model'), status_code=code, retryable=retryable, retry_not_before=self._clock() + delay if retryable else None))
            self.budget()
            if status in ('received', 'invalid_response'):
                return parsed
            if retryable:
                if retries >= self.MAX_RETRIES:
                    raise RuntimeError('API retry limit reached; inspect retry_history')
                self._schedule(job_id, delay, error)
                continue
            raise RuntimeError('Permanent or unknown API failure; inspect receipt')

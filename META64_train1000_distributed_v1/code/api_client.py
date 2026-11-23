"""Original provider protocol; hard request cap, durable receipts, no retries."""
import json,time,threading,re,shlex
from pathlib import Path
from urllib.parse import urlparse
from initskill.io import dump,digest
import portable_lock as fcntl
def read_credentials(path):
    values = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        m = re.match(r'^\s*(?:export\s+)?(MODEL_API_KEY|MODEL_API_BASE_URL)\s*=\s*(.*)$', line)
        if m:
            parts = shlex.split(m[2], comments=True)
            if len(parts) != 1:
                raise ValueError('Invalid credential setting')
            values[m[1]] = parts[0]
    if not all(values.get(k) for k in ('MODEL_API_KEY', 'MODEL_API_BASE_URL')):
        raise ValueError('Missing model credential/base URL')
    return values

class Client:
    def __init__(self,output,config,env_file=None,backend=None):
        self.output=Path(output);self.config=config;self.env_file=env_file;self.backend=backend
        self._backend_lock=threading.Lock()
    def budget(self):
        attempts=0;received=0;tokens={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
        for p in (self.output/'requests').glob('*/attempt.json'):
            attempts+=1;r=p.parent/'receipt.json'
            if r.exists():
                receipt=json.loads(r.read_text(encoding='utf-8'));received+=int(receipt['status']=='received');usage=receipt.get('usage') or {}
                for k in tokens:
                    if type(usage.get(k)) is int:tokens[k]+=usage[k]
        result=dict(attempts=attempts,received=received,max_api_calls=self.config['max_api_calls'],usage=tokens,invoice_queried=False)
        dump(self.output/'budget_status.json',result);return result
    def call(self,job_id,messages,response_format,max_tokens=8192):
            # Provider max_tokens includes both reasoning and final answer.
            payload=dict(model=self.config['model'],messages=messages,temperature=.2,
                max_tokens=self.config['reasoning_max_tokens'],response_format=response_format,
                reasoning_effort='low',extra_body={'thinking':{'type':'enabled'}})
            identity=digest(payload);folder=self.output/'requests'/job_id
            with (self.output/'.api.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if (folder/'attempt.json').exists():
                    if json.loads((folder/'attempt.json').read_text(encoding='utf-8'))['identity']!=identity:raise ValueError('Request changed')
                    if not (folder/'receipt.json').exists():raise RuntimeError('Unsettled request; no resend')
                    receipt=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
                    if receipt['status']=='transport_error':raise RuntimeError('Unknown outcome; no resend')
                    return json.loads((folder/'parsed.json').read_text(encoding='utf-8'))
                if (self.output/'PAUSE_REQUESTED').exists():raise InterruptedError('Pause requested')
                size=len(json.dumps(payload,ensure_ascii=False).encode());budget=self.budget()
                if size>self.config['max_input_bytes'] or budget['attempts']>=self.config['max_api_calls']:raise RuntimeError('Request cap')
                dump(folder/'request.json',payload)
                dump(folder/'attempt.json',dict(identity=identity,started_unix=time.time(),reserved_completion_tokens=payload['max_tokens']))
            started=time.monotonic();raw={};parsed=None;error=None;status='transport_error'
            _transient=('RateLimitError','APIConnectionError','APITimeoutError','ConnectError','ReadError','ConnectTimeout','ReadTimeout','ConnectionError','TimeoutError','RemoteProtocolError')
            try:
                for attempt in range(5):
                    try:
                        with self._backend_lock:
                            if self.backend is None:
                                from openai import OpenAI,DefaultHttpxClient
                                c=read_credentials(self.env_file)
                                if urlparse(c['MODEL_API_BASE_URL']).hostname!=self.config['provider_host']:raise ValueError('Provider changed')
                                self.backend=OpenAI(api_key=c['MODEL_API_KEY'],base_url=c['MODEL_API_BASE_URL'],max_retries=0,
                                    timeout=600,http_client=DefaultHttpxClient(follow_redirects=False))
                        response=self.backend.chat.completions.create(**payload)
                        raw=response.model_dump(mode='json');dump(folder/'response.json',raw)
                        if raw.get('model')!=self.config['model'] or response.choices[0].finish_reason!='stop':raise ValueError('Model mismatch or truncated output')
                        def nonfinite(value):raise ValueError('Nonfinite JSON constant')
                        parsed=json.loads(response.choices[0].message.content or '',parse_constant=nonfinite);status='received'
                        break
                    except Exception as exc:
                        code=getattr(exc,'status_code',None)
                        if (code==429 or type(exc).__name__ in _transient) and attempt<4:
                            time.sleep(30.0*(attempt+1))
                            continue
                        raise
            except Exception as exc:
                code=getattr(exc,'status_code',None)
                status='invalid_response' if raw else 'rejected' if code in (400,401,403,404,422,429) else 'transport_error'
                error=type(exc).__name__
                if code:dump(folder/'http_error.json',dict(status_code=code,error_type=error,message=str(getattr(exc,'message',''))[:1500]))
            dump(folder/'parsed.json',parsed)
            usage=raw.get('usage') or {};details=usage.get('completion_tokens_details') or {}
            dump(folder/'receipt.json',dict(status=status,error_type=error,usage=usage,reasoning_tokens=details.get('reasoning_tokens'),
                requested_effort='low',documented_provider_mapping='low',thinking='enabled',
                seconds=time.monotonic()-started,response_id=raw.get('id'),model=raw.get('model')))
            self.budget()
            if status in ('transport_error','rejected'):raise RuntimeError('API '+status+'; inspect receipt before continuing')
            return parsed

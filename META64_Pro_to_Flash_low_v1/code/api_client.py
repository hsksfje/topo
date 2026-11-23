"""Durable API receipts, bounded transient retries and an unchanged hard call cap."""
import json,time,threading,re,shlex,shutil
from pathlib import Path
from urllib.parse import urlparse
from email.utils import parsedate_to_datetime
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
    MAX_RETRIES=8
    def __init__(self,output,config,env_file=None,backend=None,clock=None,sleep=None):
        self.output=Path(output);self.config=config;self.env_file=env_file;self.backend=backend
        self._backend_lock=threading.Lock();self._ledger_lock=threading.RLock();self._state_lock=threading.Lock()
        self._clock=clock or time.time;self._sleep=sleep or time.sleep;self._next_start=0
        state=self.output/'api_retry_state.json'
        self._cooldown_until=json.loads(state.read_text(encoding='utf-8')).get('cooldown_until',0) if state.exists() else 0

    def budget(self):
        with self._ledger_lock:
            attempts=0;received=0;tokens={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
            # Includes archived retries: a retry is never a free or uncounted request.
            for p in (self.output/'requests').rglob('attempt.json'):
                attempts+=1;r=p.parent/'receipt.json'
                if r.exists():
                    receipt=json.loads(r.read_text(encoding='utf-8'));received+=int(receipt['status']=='received');usage=receipt.get('usage') or {}
                    for k in tokens:
                        if type(usage.get(k)) is int:tokens[k]+=usage[k]
            result=dict(attempts=attempts,received=received,max_api_calls=self.config['max_api_calls'],usage=tokens,invoice_queried=False)
            dump(self.output/'budget_status.json',result);return result

    def _check_pause(self):
        if (self.output/'PAUSE_REQUESTED').exists():raise InterruptedError('Pause requested')

    def _wait(self):
        while True:
            self._check_pause()
            with self._state_lock:
                delay=max(self._cooldown_until,self._next_start)-self._clock()
                if delay<=0:
                    self._next_start=self._clock()+10
                    return
            self._sleep(min(delay,1))

    def _retry_delay(self,error,retries):
        delay=min(900,60*2**min(retries,4))
        headers=getattr(getattr(error,'response',None),'headers',{}) or {}
        value=headers.get('retry-after') or headers.get('Retry-After')
        try:
            requested=float(value)
        except (TypeError,ValueError):
            try:requested=parsedate_to_datetime(value).timestamp()-self._clock()
            except (TypeError,ValueError,OverflowError):requested=0
        return max(delay,requested) if requested>=0 else delay

    def _schedule(self,job_id,delay,reason):
        with self._state_lock:
            self._cooldown_until=max(self._cooldown_until,self._clock()+delay)
            dump(self.output/'api_retry_state.json',dict(status='waiting_to_retry',job_id=job_id,reason=reason,
                 cooldown_until=self._cooldown_until,updated_unix=self._clock()))
        print('API_RETRY_WAIT',job_id,'seconds',round(delay),'reason',reason,flush=True)

    @staticmethod
    def _retryable(receipt,folder):
        if 'retryable' in receipt:return receipt['retryable']
        # Recognize old 429 receipts without deleting or changing them.
        err=folder/'http_error.json'
        code=json.loads(err.read_text(encoding='utf-8')).get('status_code') if err.exists() else None
        return code in (408,409,429) or isinstance(code,int) and code>=500 or receipt.get('error_type') in ('APITimeoutError','APIConnectionError','TimeoutError','ConnectionError')

    def call(self,job_id,messages,response_format,max_tokens=8192):
        payload=dict(model=self.config['model'],messages=messages,temperature=.2,
            max_tokens=self.config['reasoning_max_tokens'],response_format=response_format,
            reasoning_effort='medium',extra_body={'thinking':{'type':'enabled'}})
        identity=digest(payload);folder=self.output/'requests'/job_id
        if (folder/'attempt.json').exists() and (folder/'receipt.json').exists():
            if json.loads((folder/'attempt.json').read_text(encoding='utf-8'))['identity']!=identity:raise ValueError('Request changed')
            if json.loads((folder/'receipt.json').read_text(encoding='utf-8'))['status'] in ('received','invalid_response'):
                return json.loads((folder/'parsed.json').read_text(encoding='utf-8'))
        retries=len(list((folder/'retry_history').glob('attempt_*')))
        while True:
            self._wait()
            with self._ledger_lock, (self.output/'.api.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                attempt=folder/'attempt.json'
                if attempt.exists():
                    if json.loads(attempt.read_text(encoding='utf-8'))['identity']!=identity:raise ValueError('Request changed')
                    receipt_path=folder/'receipt.json'
                    if not receipt_path.exists():raise RuntimeError('Unsettled request; inspect outcome before resend')
                    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
                    status=receipt['status']
                    if status in ('received','invalid_response'):
                        return json.loads((folder/'parsed.json').read_text(encoding='utf-8'))
                    if not self._retryable(receipt,folder):raise RuntimeError('Permanent or unknown API failure; inspect receipt')
                    if retries>=self.MAX_RETRIES:raise RuntimeError('API retry limit reached; inspect retry_history')
                    due=receipt.get('retry_not_before',0)
                    if due>self._clock():
                        self._schedule(job_id,due-self._clock(),receipt.get('error_type'))
                        continue
                    self._check_pause()
                    if self.budget()['attempts']>=self.config['max_api_calls']:raise RuntimeError('Request cap including retries')
                    # Copy all evidence first, then replace the live attempt. An interruption here
                    # can only overcount, never erase an attempt or exceed the cap silently.
                    archive=folder/'retry_history'/('attempt_%04d'%(retries+1))
                    archive.mkdir(parents=True,exist_ok=False)
                    for name in ('request.json','attempt.json','receipt.json','parsed.json','http_error.json','response.json'):
                        if (folder/name).exists():shutil.copy2(folder/name,archive/name)
                    for name in ('attempt.json','receipt.json','parsed.json','http_error.json','response.json'):
                        (folder/name).unlink(missing_ok=True)
                    retries+=1
                self._check_pause()
                size=len(json.dumps(payload,ensure_ascii=False).encode());budget=self.budget()
                if size>self.config['max_input_bytes'] or budget['attempts']>=self.config['max_api_calls']:raise RuntimeError('Request cap')
                dump(folder/'request.json',payload)
                dump(folder/'attempt.json',dict(identity=identity,started_unix=self._clock(),reserved_completion_tokens=payload['max_tokens'],retry_number=retries))
            started=time.monotonic();raw={};parsed=None;error=None;status='transport_error';retryable=False;delay=0;code=None
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
            except Exception as exc:
                code=getattr(exc,'status_code',None);error=type(exc).__name__
                status='invalid_response' if raw else 'rejected' if code is not None else 'transport_error'
                retryable=not raw and (code in (408,409,429) or isinstance(code,int) and code>=500 or error in ('APITimeoutError','APIConnectionError','TimeoutError','ConnectionError'))
                if code:dump(folder/'http_error.json',dict(status_code=code,error_type=error,message=str(getattr(exc,'message',''))[:1500]))
                if retryable:delay=self._retry_delay(exc,retries)
            dump(folder/'parsed.json',parsed)
            usage=raw.get('usage') or {};details=usage.get('completion_tokens_details') or {}
            dump(folder/'receipt.json',dict(status=status,error_type=error,usage=usage,reasoning_tokens=details.get('reasoning_tokens'),
                requested_effort='medium',documented_provider_mapping='low',thinking='enabled',
                seconds=time.monotonic()-started,response_id=raw.get('id'),model=raw.get('model'),
                status_code=code,retryable=retryable,retry_not_before=self._clock()+delay if retryable else None))
            self.budget()
            if status in ('received','invalid_response'):return parsed
            if retryable:
                if retries>=self.MAX_RETRIES:raise RuntimeError('API retry limit reached; inspect retry_history')
                self._schedule(job_id,delay,error)
                continue
            raise RuntimeError('Permanent or unknown API failure; inspect receipt')

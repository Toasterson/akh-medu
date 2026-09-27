"""Process isolation between resident-model backends."""
import time

TERMINAL = {'completed','failed','cancelled'}


def assert_owned_turns_idle(api, row):
    owned=[(row['parent'],row['parent_turn'])]
    for session_key,turn_key in (('patch_session','patch_turn'),('review_session','review_turn')):
        if row.get(turn_key):owned.append((row[session_key],row[turn_key]))
    observations=[]
    for session,turn_id in owned:
        state=api('/v1/sessions/'+session)
        turn=next(item for item in state['turns'] if item['id']==turn_id)
        observations.append({'session':session,'turn':turn_id,'status':turn['status']})
    if any(item['status'] not in TERMINAL for item in observations):
        raise RuntimeError('backend switch refused while an owned turn is nonterminal')
    return observations


class RuntimeBoundary:
    def __init__(self, api, start, stop, initial):
        self.api,self.start,self.stop,self.current=api,start,stop,initial
        self.process=None;self.sequence=0

    def attach(self, process):
        self.process=process

    def switch(self, target, row):
        observations=assert_owned_turns_idle(self.api,row)
        if target==self.current:
            return {'schema':'lilith/runtime-boundary@1','from':self.current,'to':target,
                    'changed':False,'owned_turns':observations}
        old=self.process
        old_pid=old.pid
        self.stop(old)
        if old.poll() is None:
            raise RuntimeError('old runtime process did not stop')
        self.sequence+=1
        new=self.start(target,self.sequence)
        self.process=new;previous=self.current;self.current=target
        return {'schema':'lilith/runtime-boundary@1','from':previous,'to':target,'changed':True,
                'old_pid':old_pid,'old_exit_code':old.returncode,'new_pid':new.pid,
                'owned_turns':observations,'monotonic_seconds':time.monotonic()}

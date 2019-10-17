import numpy as np
import time

zeros = np.array([0,0,0,0,0,0,0,0], dtype=np.uint8)

try:
    import PyDAQmx
    from PyDAQmx import Task
    def setup_triggers(address="/Dev1/port0/line0:7"):
        task = Task()
        task.CreateDOChan("/Dev1/port0/line0:7","",PyDAQmx.DAQmx_Val_ChanForAllLines)
        task.StartTask()
        return task
    
    def shutdown_triggers(task):
        task.StopTask()
        
    def send_trigger_fast(val, task, fail=True):
        # print 'Sending trigger %i' % val
        try:
            data = np.array(list(np.binary_repr(val, 8))).astype(np.uint8)
            data = np.flip(data, 0).astype(np.uint8)
            task.WriteDigitalLines(1,1,10.0,PyDAQmx.DAQmx_Val_GroupByChannel, data, None, None)
        except Exception as e:
            print('Sending trigger %i' % val)
            print('ERROR!', e)
            if fail:
                raise e
    
    def send_trigger_slow(val, task, delay=.05, fail=True):
        zeros = np.array([0,0,0,0,0,0,0,0]).astype(np.uint8)
        send_trigger_fast(val, task, fail=fail)
        time.sleep(delay)
        task.WriteDigitalLines(1,1,10.0,PyDAQmx.DAQmx_Val_GroupByChannel, zeros, None, None)
except ImportError as e:
    print(e)
    print('Using dummy triggers')
    def setup_triggers(address="/Dev1/port0/line0:7"):
        pass
    
    def shutdown_triggers(task):
        pass
        
    def send_trigger_fast(val, task, fail=True):
        # print 'Sending trigger %i' % val
        pass
    
    def send_trigger_slow(val, task, delay=.05, fail=True):
        send_trigger_fast(val, task, fail=fail)
        time.sleep(delay)
        send_trigger_fast(0, task, fail=fail)

    
def send_trigger(val, task, slow=True, delay=.05, fail=True):
    if slow:
        send_trigger_slow(val, task, delay=delay, fail=fail)
    else:
        send_trigger_fast(val, task, fail=fail)

def test_triggers(task, n, delay=.1):
    for i in range(n):
        print(i)
        send_trigger_fast(i, task)
        time.sleep(delay)

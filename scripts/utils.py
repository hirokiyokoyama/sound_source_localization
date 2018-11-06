import rospy
from sound_source_localization.srv import Synchronize, SynchronizeResponse
import threading

class FrequencyMeter:
    def __init__(self, N):
        self._N = N
        self.reset()

    def reset(self):
        self._t = None
        self._count = 0

    def tap(self):
        self._count += 1
        dt = 0.
        if self._count % self._N == 0:
            if self._t is not None:
                dt = (rospy.Time.now()-self._t).to_sec()
            self._t = rospy.Time.now()
        if dt > 0:
            return self._N / dt
        return None

class Synchronizer:
    def __init__(self):
        self._sequence_local = 0
        self._sequence_remote = 0
        self._cond = threading.Condition()
        rospy.Service('/synchronize', Synchronize, self._srv_cb)
        self._srv = rospy.ServiceProxy('/bridged/synchronize', Synchronize)
        rospy.loginfo('Waiting for another robot.')
        self._srv.wait_for_service()
        
    def _srv_cb(self, req):
        with self._cond:
            self._sequence_remote = req.sequence
            self._cond.notify()
        return SynchronizeResponse(self._sequence_local)

    def next(self):
        self._sequence_local += 1
        self._srv(self._sequence_local)
        with self._cond:
            assert self._sequence_remote <= self._sequence_local, 'Something happened!'
            while self._sequence_remote < self._sequence_local:
                self._cond.wait()

    @property
    def sequence(self):
        return self._sequence_local

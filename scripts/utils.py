import rospy
from sound_source_localization.srv import Synchronize, SynchronizeResponse
import threading
import yaml
import numpy as np

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
        self._data_remote = ''
        self._cond = threading.Condition()
        rospy.Service('/synchronize', Synchronize, self._srv_cb)
        self._srv = rospy.ServiceProxy('/bridged/synchronize', Synchronize)
        rospy.loginfo('Waiting for another robot.')
        self._srv.wait_for_service()
        
    def _srv_cb(self, req):
        with self._cond:
            self._sequence_remote = req.sequence
            self._data_remote = req.data
            self._cond.notify()
        return SynchronizeResponse(self._sequence_local)

    def next(self, data):
        self._sequence_local += 1
        self._srv(self._sequence_local, yaml.dump(data))
        with self._cond:
            assert self._sequence_remote <= self._sequence_local, 'Something happened!'
            while self._sequence_remote < self._sequence_local:
                self._cond.wait()
            return yaml.load(self._data_remote)

    @property
    def sequence(self):
        return self._sequence_local

def relative_position(from_pose, to_pose):
    self_ori = np.arctan2(from_pose.orientation.z, from_pose.orientation.w)*2
    dy = to_pose.position.y - from_pose.position.y
    dx = to_pose.position.x - from_pose.position.x
    other_dir = np.arctan2(dy, dx)
    
    return np.hypot(dx,dy), other_dir-self_ori

def axis_vector(q, axis):
    from tf import transformations
    return transformations.quaternion_matrix([q.x,q.y,q.z,q.w])[:3,axis]

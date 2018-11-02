import rospy

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

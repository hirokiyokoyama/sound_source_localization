#!/usr/bin/env python

import websocket
import json

msg0 = """{
  "op": "advertise",
  "topic": "/remote_hsrb/laser_2d_pose",
  "type": "geometry_msgs/PoseWithCovarianceStamped"
}"""
msg1 = """{
  "op": "subscribe",
  "topic": "/laser_2d_pose",
  "type": "geometry_msgs/PoseWithCovarianceStamped"
}"""
msg2 = """{
  "op": "advertise_service",
  "type": "rosapi/TopicsForType",
  "service": "/remote_hsrb/rosapi/topics_for_type"
}"""
def on_open(ws):
    ws.send(msg1)
    ws.send(msg2)
 
def on_message(ws, message):
    message = json.loads(message)
    if message['op'] == 'call_service':
      message["service"] = message["service"][len('/remote_hsrb'):]
      message = json.dumps(message)
      ws_local.send(message)
      msg = ws_local.recv()
      msg = json.loads(msg)
      msg["service"] = '/remote_hsrb' + msg["service"]
      msg = json.dumps(msg)
      ws.send(msg)
    else:
      message["topic"] = '/remote_hsrb' + message["topic"]
      message = json.dumps(message)
      ws_local.send(message)
 
if __name__ == "__main__":
    import rospy
    rospy.init_node('ssl_bridge')
    remote_hostname = rospy.get_param('~remote_hsrb_hostname')
    #websocket.enableTrace(True)
    websocket.enableTrace(False)
    ws_local = websocket.create_connection("ws://localhost:9090")
    ws_local.send(msg0)
    ws_remote = websocket.WebSocketApp("ws://{}:9090".format(remote_hostname),
                                       on_open = on_open,
                                       on_message = on_message)
    ws_remote.run_forever()

